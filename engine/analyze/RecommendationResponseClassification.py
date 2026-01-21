import os
from enum import Enum

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from engine.utils.AICaller import ai_caller

tqdm.pandas()


class ResponseCategory(str, Enum):
    ACCEPTED_AND_IMPLEMENTED = "Accepted and Implemented"
    ACCEPTED = "Accepted"
    UNDER_CONSIDERATION = "Under consideration"
    REJECTED = "Rejected"
    NA = "N/A"


# Mapping descriptions
CATEGORY_DESCRIPTIONS = {
    "Accepted and Implemented": "The recommendation was accepted (wholly) and has been implemented",
    "Accepted": "The recommendation was accepted and is being, or will be implemented",
    "Under consideration": "The recipient has acknowledged the recommendation and will consider it",
    "Rejected": "The recommendation will not be implemented",
    "N/A": "Not applicable, redirecting to a different department or needs more information",
}


class ResponseCategoryOutput(BaseModel):
    category: ResponseCategory
    description: str

    @classmethod
    def create(cls, category: ResponseCategory):
        return cls(category=category, description=CATEGORY_DESCRIPTIONS[category])


class RecommendationResponseClassifier:
    """
    This class will take a response to a recommendation and provide a string classification of it. There are going to be 4 classifications ranging from Rejected to Accepted and Implemented.
    """

    def __init__(self):
        pass

    def classify_response(self, response, recommendation, recommendation_num):
        system_prompt = """
    You are helping me put responses into categories.

    These responses are to recommendations that were made in a transport accident investigation report. These recommendations are issued directly to a particular party. 
    
    You should respond with one of 4 categories, take note which one fits best.
    {CATEGORY_DESCRIPTIONS}

    However if there are responses that don't fit into any of the categories then you can put them as N/A. These may be responses that request further information or want recommendation to be sent elsewhere.

    Your response should just be the name of the category with nothing else.
    """
        user_prompt = f"""
    Which category is this response in?

    "
    {response}
    "

    in regards to recommendation {recommendation_num}:

    {recommendation}
    """

        openai_response = ai_caller.query(
            system_prompt,
            user_prompt,
            model="gpt-4",
            temp=0,
            output_structure=ResponseCategoryOutput,
        )
        print(
            f"Classified recommendation {recommendation_num} response as: {openai_response}"
        )

        return openai_response.category.value.lower()


class RecommendationResponseClassificationProcessor:
    """
    This class uses `RecommendationResponseClassifier` to classify the responses of the recommendations.
    """

    def __init__(self):
        self.recommendation_response_classifier = RecommendationResponseClassifier()

        self.required_columns = [
            "recommendation_id",
            "recommendation",
            "reply_text",
            "response_category",
        ]

    def process(self, input_path: str, output_path: str):
        """
        This will read the DataFrame of recommendations from the data folder and then add a response category column and a response category quality column then save the DataFrame in the output folder.
        """
        print(
            "============================================================================================================================="
        )
        print(
            "                                         ...Classifying recommendations..."
        )
        print(
            "============================================================================================================================="
        )

        if not os.path.exists(input_path):
            raise ValueError(f"{input_path} does not exist")

        # Load the data
        recommendations_df = pd.read_pickle(input_path)
        recommendations_df = pd.concat(
            [
                df.assign(report_id=id)
                for id, df in recommendations_df[["report_id", "recommendations"]]
                .dropna()
                .itertuples(index=False)
            ],
            ignore_index=True,
        ).assign(response_category=None)

        # Check to make sure it has all of the required columns
        if not all(
            [column in recommendations_df.columns for column in self.required_columns]
        ):
            missing_columns = [
                column
                for column in self.required_columns
                if column not in recommendations_df.columns
            ]
            raise RuntimeError(
                f"Required column/s {missing_columns} not found in DataFrame given\nPath of DataFrame file: {input_path}\nDataFrame columns: {list(recommendations_df.columns)}"
            )

        # Drop without replyText
        recommendations_df = recommendations_df[
            recommendations_df["reply_text"].notna()
        ]

        columns = [
            "report_id",
            "recommendation_id",
            "recipient",
            "made",
            "recommendation",
            "recommendation_context",
            "reply_text",
        ]

        # Check to see the for previously classified responses
        if os.path.exists(output_path):
            output_df = pd.read_pickle(output_path)
            merged_df = pd.merge(
                recommendations_df[columns], output_df, on=columns, how="outer"
            )
            merged_df = merged_df.drop_duplicates(subset=columns)
        else:
            merged_df = recommendations_df

        recommendations_df = self._process(merged_df)

        recommendations_df["response_category"] = recommendations_df[
            "response_category"
        ].apply(lambda x: x.lower() if isinstance(x, str) else "Classification Error")

        recommendations_df.to_pickle(output_path)

    def _process(self, recommendations: pd.DataFrame) -> pd.DataFrame:
        """
        Take a long form DataFrame of recommendations and provide a response_category column. There will be a response_category quality column
        """
        # Splits into two DataFrames based on whether response_category is already filled out.
        unclassified_responses = recommendations[
            recommendations["response_category"].isnull()
        ]
        classified_responses = recommendations[
            ~recommendations["response_category"].isnull()
        ]

        # Add response_category quality column if it doesnt exist
        if "response_category_quality" not in classified_responses.columns:
            classified_responses["response_category_quality"] = None

        classified_responses["response_category_quality"] = classified_responses[
            "response_category_quality"
        ].apply(lambda x: "exact" if x is None else x)

        print(
            f" Out of all {len(recommendations)} recommendations, {len(unclassified_responses)} need to be classified"
        )

        if len(unclassified_responses) == 0:
            return recommendations

        # For all empty response_category infer the response category

        unclassified_responses["response_category"] = (
            unclassified_responses.progress_apply(
                lambda x: self.recommendation_response_classifier.classify_response(
                    x["reply_text"], x["recommendation"], x["recommendation_id"]
                ),
                axis=1,
            )
        )

        unclassified_responses["response_category_quality"] = "inferred"

        return pd.concat(
            [classified_responses, unclassified_responses], ignore_index=True
        )
