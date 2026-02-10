"""Tests for report extraction functionality.

This module contains comprehensive tests for the ReportExtracting module,
including tests for safety issue extraction, recommendation extraction,
report chunking, and parallel processing of multiple reports.

This is the key test module for validating that the AI extraction works.
"""

import os
from difflib import SequenceMatcher
from unittest.mock import patch

import pandas as pd
import pytest

from engine.extract.ReportExtracting import (
    RecommendationItem,
    SafetyIssueItem,
    ai_read_report,
    chunk_report_into_sections,
    process_reports_parallel,
)


def get_report_text(report_id: str) -> str:
    """Fetch the text of a report.

    Args:
        report_id (str): The unique identifier for the report.

    Raises:
        ValueError: If the report ID is not found in the extracted reports.

    Returns:
        str: The text content of the report.

    """
    extracted_reports = pd.read_pickle(
        os.path.join(
            pytest.output_config["folder_name"],
            pytest.output_config["parsed_reports_df_file_name"],
        )
    )
    try:
        return extracted_reports.loc[report_id]["text"]
    except KeyError:
        raise ValueError(f"Report ID {report_id} not found in extracted reports.")


class TestAIExtraction:
    """Tests AI extraction for safety issues and recommendations.

    This class is the key class that provides some rigor to the extraction methods. It should be expanded yet currently given AI capabilities it seems reasonable that with these few tests we can expect fair generalization.
    """

    @pytest.mark.parametrize(
        "report_id, expected",
        [
            pytest.param(
                "TAIC_m_2004_203",
                [
                    SafetyIssueItem(
                        safety_issue="securing of vehicular cargo on car and rail decks",
                        quality="exact",
                    ),
                    SafetyIssueItem(
                        safety_issue="securing of heavy items of equipment in passenger accessible areas",
                        quality="exact",
                    ),
                ],
                id="safety issues are short and at the top (but no where else)",
            ),
            pytest.param(
                "TAIC_a_2020_003",
                [],
                id="no safety issues present, this is explicitly stated",
            ),
            pytest.param(
                "TAIC_m_2023_204",
                [],
                id="no safety issues present, this is explicitly stated 2",
            ),
            # TSB
            pytest.param(
                "TSB_a_2023_W0096",
                [
                    SafetyIssueItem(
                        safety_issue="If spin training is initiated from a height that does not provide a pilot a wide enough recovery margin, there is an increased risk of collision with terrain.",
                        quality="inferred",
                    ),
                    SafetyIssueItem(
                        safety_issue="If emergency locator transmitters are incorrectly installed and/or tested, they may not   function as designed, increasing the risk that search and rescue efforts are not timely.",
                        quality="inferred",
                    ),
                ],
                id="TSB report with inferred safety issues from findings",
            ),
            pytest.param("TSB_a_2023_W0096", []),
            pytest.param(
                "ATSB_a_2005_912",
                [
                    SafetyIssueItem(
                        safety_issue="The current overhaul manual provides no guidance to assemblers on the maximum allowable assembly time following the tower shaft/bevel gear heating and cooling process to reduce the likelihood of producing surface damage (galling damage) to the components during assembly.",
                        quality="exact",
                    ),
                ],
            ),
        ],
    )
    def test_safety_issue_extraction(
        self, report_id: str, expected: list[SafetyIssueItem]
    ):
        """Test the extraction of safety issues from a report.

        Test both TSB and TAIC as well as pre 2008 ATSb reports.

        Args:
            report_id (str): The unique identifier for the report.
            expected (list[SafetyIssueItem]): The expected safety issues.

        Raises:
            ValueError: If the report ID is not found in the extracted reports.

        """
        report_text = get_report_text(report_id)

        extracted_data = ai_read_report(
            agency_name=report_id.split("_")[0],
            report_text=report_text,
            safety_issues=True,
            recommendations=False,
        )

        assert len(extracted_data.safety_issues) == len(expected)

        for extracted_item, expected_item in zip(
            extracted_data.safety_issues, expected
        ):
            similarity = SequenceMatcher(
                None,
                extracted_item.safety_issue.lower(),
                expected_item.safety_issue.lower(),
            ).ratio()

            if expected_item.quality == "exact":
                assert (
                    similarity >= 0.95
                ), f"Expected near perfect match but got similarity {similarity:.2f}"
            elif expected_item.quality == "inferred":
                assert (
                    similarity >= 0.7
                ), f"Expected similar match but got similarity {similarity:.2f}"
            else:
                raise ValueError(f"Unknown quality level: {expected_item.quality}")

    @pytest.mark.parametrize(
        "report_id, expected",
        [
            pytest.param(
                "ATSB_a_2000_157",
                [
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Federal Aviation Administration (Piston Engine Certification Directorate) review the certification requirements of piston engines with respect to the operating conditions under which combustion chamber deposits that may cause preignition are formed.",
                        recommendation_id="R20010254",
                        recipient="Federal Aviation Administration (Piston Engine Certification Directorate)",
                        recommendation_context="High power variants of horizontally opposed, six-cylinder, air-cooled reciprocating engines power many aircraft employed in low capacity public transport operations in Australia. At the time of publication of this report, there were 107 Piper Chieftains on the Australian aircraft register and a much greater number in operation worldwide. Many other single and multi engine aircraft are equipped with high-powered reciprocating engines. The engine failure analysis presented in this report highlighted a number of issues that affect the reliability of these engines. Accordingly, the following recommendations are issued:",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Federal Aviation Administration FAA (Piston Engine Certification Directorate) review the practice during assembly of applying anti-galling compounds to the backs of connecting rod bearing inserts with respect to its affect on the safety margin for engine operation of the bearing insert retention forces achieved.",
                        recommendation_id="R20010255",
                        recipient="Federal Aviation Administration FAA (Piston Engine Certification Directorate)",
                        recommendation_context="High power variants of horizontally opposed, six-cylinder, air-cooled reciprocating engines power many aircraft employed in low capacity public transport operations in Australia. At the time of publication of this report, there were 107 Piper Chieftains on the Australian aircraft register and a much greater number in operation worldwide. Many other single and multi engine aircraft are equipped with high-powered reciprocating engines. The engine failure analysis presented in this report highlighted a number of issues that affect the reliability of these engines. Accordingly, the following recommendations are issued:",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that T extron Lycoming review the practice during assembly of applying anti-galling compounds to the backs of connecting rod bearing inserts with respect to its affect on the safety margin for engine operation of the bearing insert retention forces achieved during assembly.",
                        recommendation_id="R20010256",
                        recipient="T extron Lycoming",
                        recommendation_context="High power variants of horizontally opposed, six-cylinder, air-cooled reciprocating engines power many aircraft employed in low capacity public transport operations in Australia. At the time of publication of this report, there were 107 Piper Chieftains on the Australian aircraft register and a much greater number in operation worldwide. Many other single and multi engine aircraft are equipped with high-powered reciprocating engines. The engine failure analysis presented in this report highlighted a number of issues that affect the reliability of these engines. Accordingly, the following recommendations are issued:",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Civil Aviation Safety Authority review the operating and maintenance procedures for high-powered piston engines fitted to Australian registered aircraft to ensure adequate management and control of combustion chamber deposits, preignition and detonation.",
                        recommendation_id="R20010257",
                        recipient="Civil Aviation Safety Authority",
                        recommendation_context="High power variants of horizontally opposed, six-cylinder, air-cooled reciprocating engines power many aircraft employed in low capacity public transport operations in Australia. At the time of publication of this report, there were 107 Piper Chieftains on the Australian aircraft register and a much greater number in operation worldwide. Many other single and multi engine aircraft are equipped with high-powered reciprocating engines. The engine failure analysis presented in this report highlighted a number of issues that affect the reliability of these engines. Accordingly, the following recommendations are issued:",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Civil Aviation Safety Authority alert operators of aircraft equipped with turbo-charged engines to the potential risks of engine damage associated with detonation, and encourage the adoption of conservative fuel mixture leaning practices.",
                        recommendation_id="R20000250",
                        recipient="Civil Aviation Safety Authority",
                        recommendation_context="On 30 October 2000, the ATSB issued the following recommendation:",
                        made="30 October 2000",
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Civil Aviation Safety Authority educate industry on procedures and techniques that may maximise the chances of survival of a ditching event. Part of that education program should include the development of formal guidance material of the type contained in the UK CAA General Aviation Safety Senses leaﬂet 21A Ditching.",
                        recommendation_id="R20010258",
                        recipient="Civil Aviation Safety Authority",
                        recommendation_context="The report includes a recommendation to CASA regarding guidance material for pilots on ditching.",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Civil Aviation Safety Authority amend Civil Aviation Order Section 20.11 paragraph 5.1.2 to remove the restriction that it only applies to aircraft authorised to carry more than nine passengers.",
                        recommendation_id="R20000248",
                        recipient="Civil Aviation Safety Authority",
                        recommendation_context="On 30 October 2000, the ATSB issued the following recommendation:",
                        made="30 October 2000",
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Civil Aviation Safety Authority ensure that Civil Aviation Orders provide for adequate emergency and life saving equipment for the protection of fare-paying passengers during over-water flights where an aircraft is operating beyond the distance from which it could reach the shore with all engines inoperative.",
                        recommendation_id="R20000249",
                        recipient="Civil Aviation Safety Authority",
                        recommendation_context="On 30 October 2000, the ATSB issued the following recommendation:",
                        made="30 October 2000",
                    ),
                ],
                id="multiple recommendations present, old format, old one to be ignored",
            ),
            pytest.param(
                "ATSB_a_2020_007",
                [
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the New South Wales Rural Fire Service take further action to address the absence of policies and procedures for personnel to effectively manage and communicate task rejections on the basis of operational safety concerns.",
                        recommendation_id="AO-2020-007-SR-09",
                        recipient="NSW Rural Fire Service",
                        recommendation_context="While the ATSB acknowledges the commitment to undertake reviews and research, at the time of publication, the New South Wales Rural Fire Service had not yet committed to any safety action that would reduce the risk associated with the identified safety issue to an acceptable level. The RFS outlined to the ATSB during the course of the investigation that they had access to the US policies and procedures relating to the use of large air tankers and had referred to these in developing and managing the large air tanker program. Noting that the RFS are closely involved in aerial operations, the ATSB considers that the inclusion of policies and procedures for task rejections would provide RFS personnel with the necessary information to effectively manage and communicate taskings on the basis of safety. As such, the ATSB issues the following safety recommendation to the New South Wales Rural Fire Service to take further action to address this safety issue",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the New South Wales Rural Fire Service take further action to address the absence of policies and procedures regarding minimum aerial supervision requirements and the use of initial attack to assist frontline staff with making acceptable risk-based tasking decisions.",
                        recommendation_id="AO-2020-007-SR-10",
                        recipient="NSW Rural Fire Service",
                        recommendation_context="While the ATSB acknowledges the commitment to undertake various reviews and research, at time of publication, the New South Wales Rural Fire Service had not yet committed to any safety action that would reduce the risk associated with the identified safety issue to an acceptable level. The RFS outlined to the ATSB during the course of the investigation that they had access to the US policies and procedures relating to the use of large air tankers and had referred to these in developing and managing the large air tanker program. Policies and procedures regarding aerial supervision and the use of initial attack would ensure taskings can be conducted within their defined, accepted risk levels considering the elevated risks associated with such taskings. As such, the ATSB issues the following safety recommendation to the New South Wales Rural Fire Service to take further action to address this safety issue.",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the New South Wales Rural Fire Service address the ambiguity with the interpretation of 'initial attack' in the NSW and ACT Aviation Standard Operating Procedures with the intent of this requirement.",
                        recommendation_id="AO-2020-007-SR-08",
                        recipient="NSW Rural Fire Service",
                        recommendation_context="While the ATSB notes the intention to conduct an audit, undertake a review and conduct further research, it is uncertain how these proposed safety actions will remove the ambiguity between the procedure and intention for pilots to have the United States Department of Agriculture Forest Service (USFS) initial attack certification, first notified to the NSW RFS in July 2020. As such, there is no assurance that crews operating as initial attack will be consistently certified to the same requirements as that achieved through the USFS certification process. Therefore, the ATSB issues the following safety recommendation to the New South Wales Rural Fire Service to take further action to address this safety issue.",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that Coulson Aviation further consider the fitment of a windshear detection system to their C-130 aircraft to minimise the time taken for crews to recognise and respond to an encounter particularly when operating at low-level and low speed.",
                        recommendation_id="AO-2020-007-SR-11",
                        recipient="Coulson Aviation",
                        recommendation_context="The ATSB acknowledges that forward-looking (predictive) windshear warning systems may have reduced effectiveness in drier environments. However, on the day of the accident, the windshear system on the Boeing 737 (whether predictive or reactive) activated when operating in similar environmental conditions to that very likely experienced by N134CG. Further, the fitment of a windshear system in the Lockheed Martin 'FireHerc', and acknowledgement by some of the pilots interviewed that they have had a positive effect on managing a windshear encounter, would suggest that these systems have a degree of effectiveness. In the absence of an airborne detection system, a successful recovery from a windshear encounter is reliant on the pilot's timely recognition and response, which research has shown could take 5 to 15 seconds. When conducting firefighting operations in the low-level environment, there is often limited time and height available for such recognition and response. Therefore, the ATSB believes that the fitment of windshear detection systems to the C-130 aircraft would be an important safety enhancement for aerial firefighting operations. As such, the ATSB issues the following safety recommendation to Coulson Aviation to take further action to address this safety issue",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that Coulson Aviation take further action to incorporate foreseeable external factors into their pre-flight assessment tool to ensure the overall risk profile of a tasking can be consistently assessed by crews.",
                        recommendation_id="AO-2020-007-SR-12",
                        recipient="Coulson Aviation",
                        recommendation_context="The ATSB acknowledges that Coulson Aviation introduced a pre-flight risk assessment tool into their fixed-wing operations. However, this tool did not consider all foreseeable external factors that could elevate the risk of a flight, such as weather-related task rejections or cancellations by others, which was considered the highest risk factor by the Helicopter Association International. Without this factor, the overall risk profile for a tasking, or essentially the safety of the flight, could be underestimated by crews. Therefore, as it is critical that crews can differentiate between a low-risk and high-risk flight in the already elevated risk environment of aerial firefighting, the ATSB issues the following safety recommendation to Coulson Aviation to take further action to reduce the risk of this safety issue to low as reasonably practicable.",
                        made=None,
                    ),
                ],
                id="recommendations new format",
            ),
            pytest.param(
                "ATSB_r_2014_001", [], id="recommendations present 2014 format"
            ),
            pytest.param(
                "ATSB_m_2007_241",
                [
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Tasmanian Ports Corporation takes action to address this safety issue.",
                        recommendation_id="MR20090001",
                        recipient="Tasmanian Ports Corporation",
                        recommendation_context="The pilot had received no training in how to use anchors to assist him with handling a ship in the confined approaches to Grassy. As a result, he was not comfortable with using the ship's anchors to assist him with manoeuvring the ship in the close confines of the approaches to the inner harbour.",
                    )
                ],
                id="marime old format",
            ),
            pytest.param(
                "ATSB_m_2022_007",
                [
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that Lloyd's Register takes steps to approach the International Association of Classification Societies and seek safety action to address the risk associated with a single point of failure in electrical power supply for ship's rudder angle indicators.",
                        recommendation_id="MO-2022-007-SR-34",
                        recipient="Lloyd's Register",
                        recommendation_context="The ATSB acknowledges that alternate design solutions may effectively address the risk associated with a single point of failure in electrical power supply for ship's rudder angle indicators and that this may take the form of a unified interpretation (UI) of SOLAS requirements. The ATSB does not prescribe the form that corrective action to address the safety issue should take and it is up to the responsible organisation(s) to identify the most effective means of addressing the risk. In the absence of a detailed proposal and/or timeframe to approach IACS, raise this safety issue and best ensure safety action that will address the risk, the ATSB issues the following safety recommendation.",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Australian Maritime Safety Authority provides the necessary support and assistance to the Liberia Maritime Authority in its efforts to seek safety action at the International Maritime Organization aimed at addressing the risk associated with a single point of failure in electrical power supply for ship's rudder angle indicators.",
                        recommendation_id="MO-2022-007-SR-35",
                        recipient="Australian Maritime Safety Authority",
                        recommendation_context="The ATSB notes AMSA's position that it will consider any requests for support from the Liberian Administration in progressing the safety issue and achieving safety action at the IMO. The safety issue was directed, both to AMSA and the Liberia Maritime Authority (LiMA), to best ensure safety action to address the safety issue. In addition, the ATSB will use its membership in the IMO's Casualty Analysis Correspondence Group (CACG) that reports to the Sub-Committee on Implementation of IMO Instruments (III) to highlight the safety issue and seek like-minded support. In the interim, to support AMSA in providing the Liberian Administration with the necessary assistance, the ATSB issues the following safety recommendation.",
                        made=None,
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Liberia Maritime Authority takes steps to formally raise this safety issue with the International Maritime Organization to seek safety action aimed at addressing the risk associated with a single point of failure in electrical power supply for ship's rudder angle indicators.",
                        recommendation_id="MO-2022-007-SR-36",
                        recipient="Liberia Maritime Authority",
                        recommendation_context="ATSB comment\n\nThe ATSB acknowledges the Liberia Maritime Authority's advice that the issuance of a marine advisory is being considered and that the safety issue is being addressed with the ship's classification society, Lloyd's Register, and with IACS. However, in the absence of a detailed proposal and/or timeframe for action from Lloyd's Register and IACS or of a proposal to raise this issue with the International Maritime Organization through the appropriate process and best ensure safety action aimed at addressing this issue, the ATSB issues the following safety recommendation.",
                        made=None,
                    ),
                ],
                id="marime new format, context far away",
            ),
            pytest.param(
                "ATSB_r_2010_007",
                [
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that the Australian Rail Track Corporation take action to address this safety issue.",
                        recommendation_id="RO-2010-SR-006",
                        recipient="Australian Rail Track Corporation",
                        recommendation_context="Rule ANWT 304 does not stipulate that the Protection Officer must inform all persons or work groups who may be within the boundaries of a Track Occupancy Authority of its existence. This is regardless of whether or not these persons or work groups fit the definition of 'work parties' or 'workers'.",
                    ),
                    RecommendationItem(
                        recommendation="The Australian Transport Safety Bureau recommends that RailCorp take action to address this safety issue.",
                        recommendation_id="RO-2010-SR-007",
                        recommendation_context="Rule ANWT 304 does not stipulate that the Protection Officer must inform all persons or work groups who may be within the boundaries of a Track Occupancy Authority of its existence. This is regardless of whether or not these persons or work groups fit the definition of 'work parties' or 'workers'.",
                        recipient="RailCorp",
                    ),
                ],
                id="rail safety bulletin format",
            ),
            # adverserial tests
            pytest.param(
                "ATSB_a_2007_018",
                [],
                id="previous recommendation listed yet not present in report text",
            ),
        ],
    )
    def test_recommendation_extraction(
        self, report_id: str, expected: list[RecommendationItem]
    ):
        """Test the recommendation extraction of ATSB.

        Args:
            report_id (str): The unique identifier for the report.
            expected (list[RecommendationItem]): The expected recommendations.

        """
        report_text = get_report_text(report_id)

        extracted_data = ai_read_report(
            agency_name=report_id.split("_")[0],
            report_text=report_text,
            safety_issues=False,
            recommendations=True,
        )

        print(extracted_data.recommendations)

        assert len(extracted_data.recommendations) == len(expected)

        for extracted_item, expected_item in zip(
            extracted_data.recommendations, expected
        ):
            similarity = SequenceMatcher(
                None,
                extracted_item.recommendation.lower(),
                expected_item.recommendation.lower(),
            ).ratio()

            assert (
                similarity >= 0.95
            ), f"Expected near perfect match but got similarity {similarity:.2f}"

            assert extracted_item.recommendation_id == expected_item.recommendation_id

            assert extracted_item.recipient == expected_item.recipient

            context_similarity = SequenceMatcher(
                None,
                (extracted_item.recommendation_context or "").lower(),
                (expected_item.recommendation_context or "").lower(),
            ).ratio()

            assert (
                context_similarity >= 0.9
            ), f"Expected near perfect context match but got similarity {context_similarity:.2f}"


@pytest.mark.parametrize(
    "report_id, num_sections",
    [
        pytest.param(
            "TAIC_m_2004_203",
            37,
        ),
        pytest.param(
            "TSB_a_2023_W0096",
            42,
        ),
        pytest.param(
            "TAIC_a_2020_003",
            69,
        ),
        pytest.param(
            "ATSB_r_2010_007",
            22,
        ),
    ],
)
def test_chunking_into_section(report_id, num_sections):
    """Test that does a basic sanity check on the chunking of a report into sections."""
    report_text = get_report_text(report_id)

    sections = chunk_report_into_sections(report_text)

    assert (
        len(sections) == num_sections
    ), f"Expected {num_sections} sections but got {len(sections)}"


def test_parallel_extraction():
    """Test that we can extract from multiple reports in parallel without issues."""
    ids = {
        "ATSB_a_2000_157": {
            "si": 7,
            "recs": 8,
            "sections": 217,  # Estimated
        },
        "ATSB_r_2014_001": {
            "si": 0,
            "recs": 2,
            "sections": 134,  # Estimated
        },
        "ATSB_a_2007_018": {
            "si": 4,
            "recs": 0,
            "sections": 65,  # Estimated
        },
        "ATSB_m_2007_241": {
            "si": 4,
            "recs": 1,
            "sections": 60,  # Estimated
        },
        "ATSB_m_2022_007": {
            "si": 0,
            "recs": 3,
            "sections": 136,  # Estimated
        },
        "ATSB_r_2010_007": {
            "si": 0,
            "recs": 2,
            "sections": 22,
        },
        "TAIC_a_2020_003": {
            "si": 0,
            "recs": 0,
            "sections": 69,
        },
        "TSB_a_2023_W0096": {
            "si": 2,
            "recs": 0,
            "sections": 42,
        },
        "ATSB_a_2005_912": {
            "si": 1,
            "recs": 0,
            "sections": 34,  # Estimated
        },
    }

    # These extra ids will be used to make sure that it can ignore reports that have already been processed

    test_report_texts = [get_report_text(report_id) for report_id in list(ids.keys())]

    report_texts_df = pd.DataFrame(
        {"report_id": list(ids.keys()), "text": test_report_texts}
    )

    # Extract from all reports in parallel

    results = process_reports_parallel(
        reports_df=report_texts_df,
        current_extracted_df=None,
    )

    # Compare results to expected values
    failures = []
    for report_id, expected in ids.items():
        extracted = results[results["report_id"] == report_id].iloc[0]

        if len(extracted.safety_issues) != expected["si"]:
            failures.append(
                f"{report_id}: safety_issues expected {expected['si']}, got {len(extracted.safety_issues)}"
            )
        if len(extracted.recommendations) != expected["recs"]:
            failures.append(
                f"{report_id}: recommendations expected {expected['recs']}, got {len(extracted.recommendations)}"
            )
        if len(extracted.sections) != expected["sections"]:
            failures.append(
                f"{report_id}: sections expected {expected['sections']}, got {len(extracted.sections)}"
            )

    assert not failures, "Extraction mismatches:\n  " + "\n  ".join(failures)


def test_process_handle_already_processed():
    """Test that process_reports_parallel only processes new reports and skips already processed ones."""
    already_processed_ids = [
        "ATSB_r_2014_001",
        "ATSB_a_2007_018",
        "ATSB_m_2007_241",
        "ATSB_m_2022_007",
    ]

    # IDs to process: one is a duplicate (already processed), two are new
    to_process_ids = [
        "ATSB_a_2007_018",  # Duplicate - should NOT be re-processed
        "ATSB_m_2020_007",  # New - should be processed
        "TAIC_m_2004_203",  # New - should be processed
    ]

    # Expected new report IDs (duplicates removed)
    expected_new_ids = ["ATSB_m_2020_007", "TAIC_m_2004_203"]

    # Mock up a current extracted df with some already processed reports
    current_extracted_df = pd.DataFrame(
        {
            "report_id": already_processed_ids,
            "safety_issues": [[] for _ in already_processed_ids],
            "recommendations": [[] for _ in already_processed_ids],
            "sections": [[] for _ in already_processed_ids],
        }
    )

    # Create a reports df with both already processed and new reports
    report_texts_df = pd.DataFrame(
        {
            "report_id": already_processed_ids + to_process_ids,
            "text": ["dummy_text"] * (len(already_processed_ids) + len(to_process_ids)),
        }
    )

    # Mock the extract_report function to track calls
    with patch("engine.extract.ReportExtracting.extract_report") as mock_extract:
        # Configure mock to return a dict with expected structure
        mock_extract.side_effect = lambda row: {
            "report_id": row["report_id"],
            "safety_issues": [],
            "recommendations": [],
            "sections": {},
        }

        # Process reports
        results_df = process_reports_parallel(
            reports_df=report_texts_df,
            current_extracted_df=current_extracted_df,
        )

    # Assertions: Verify that extract_report was called only for new reports
    assert (
        mock_extract.call_count == 2
    ), f"extract_report should be called 2 times (for new reports), but was called {mock_extract.call_count} times"

    # Verify the correct reports were processed by checking the call arguments
    called_report_ids = [
        call_arg[0][0]["report_id"] for call_arg in mock_extract.call_args_list
    ]
    assert (
        set(called_report_ids) == set(expected_new_ids)
    ), f"extract_report should be called for {expected_new_ids}, but was called for {called_report_ids}"

    # Assertions: Verify results contain all reports (already processed + new)
    result_ids = set(results_df["report_id"].tolist())
    expected_all_ids = set(already_processed_ids + expected_new_ids)
    assert (
        result_ids == expected_all_ids
    ), f"Results should contain {expected_all_ids}, but contains {result_ids}"

    # Assertions: Verify that all required columns are present
    assert (
        set(results_df.columns)
        == {"report_id", "safety_issues", "recommendations", "sections"}
    ), f"Results should have columns ['report_id', 'safety_issues', 'recommendations', 'sections'], but has {list(results_df.columns)}"

    # Assertions: No duplicate report_ids in results
    assert (
        len(results_df) == len(result_ids)
    ), f"Results should have {len(expected_all_ids)} unique reports, but has {len(results_df)} rows"
