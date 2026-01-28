import math
import os
from difflib import SequenceMatcher

import pandas as pd
import pytest

from engine.extract.ReportExtracting import (
    RecommendationsExtractor,
    ReportExtractor,
    ReportSectionExtractor,
    SafetyIssueExtractor,
)


@pytest.mark.parametrize(
    "report_id, expected",
    [
        # TAIC reports
        pytest.param(
            "TAIC_m_2016_204",
            ["Abbreviations ii  \nG", "ine Safety Code 40  ", 1395],
            id="TAIC_m_2016_204",
        ),
        pytest.param(
            "TAIC_r_2002_122",
            ["\nAbbreviations iii", "Appendix 1 28", 1180],
            id="TAIC_r_2002_122",
        ),
        pytest.param(
            "TAIC_a_2010_001",
            ["\nAbbreviations  vii  ", " December 2009  14  ", 637],
            id="TAIC_a_2010_001",
        ),
        pytest.param(
            "TAIC_m_2020_202",
            ["1 - Executive summar", " the Commission 35  ", 1555],
            id="TAIC_m_2020_202",
        ),
        pytest.param(
            "TAIC_r_2019_106",
            ["1 - Executive summar", "mission reports 19  ", 897],
            id="TAIC_r_2019_106",
        ),
        pytest.param(
            "TAIC_a_2018_006",
            ["1 - Executive summar", " 40  \nCargo pod 40  ", 1438],
            id="TAIC_a_2018_006",
        ),
        pytest.param(
            "TAIC_m_2010_204",
            ["Abbreviations ii", "e Hanjin Bombay 31  ", 1100],
            id="TAIC_m_2010_204",
        ),
        # ATSB reports
        pytest.param(
            "ATSB_m_2000_157",
            ["\nSummary 1  \nSources of information ", "Details of ship 33", 619],
            id="ATSB_m_2000_157 (spaces in the dots)",
        ),
        pytest.param(
            "ATSB_a_2007_012",
            ["i - Table of Contents", "IRS Mode Selector Unit 84", 3695],
            id="ATSB_a_2007_012 (long content section)",
        ),
        pytest.param(
            "ATSB_a_2023_011",
            None,
            id="ATSB_a_2023_011 (No content section)",
        ),
        pytest.param(
            "ATSB_m_2001_170",
            ["Summary 1  \nS", " - Attachment 1 25  ", 470],
            id="ATSB_m_2001_170 (discarding matches outside of content section)",
        ),
        pytest.param(
            "ATSB_r_2021_002",
            ["Safety summary 3  ", "Submissions 30", 3000],
            id="ATSB_r_2021_002 (Long content section)",
        ),
        pytest.param(
            "TSB_r_2020_V0230",
            ["Rail Transportation ", "- - Safety message 7", 455],
            id="TSB_r_2020_V0230 (Using the pdf headers)",
        ),
    ],
)
def test_content_section_extraction(report_id, expected):
    extracted_reports = pd.read_pickle(
        os.path.join(
            pytest.output_config["folder_name"],
            pytest.output_config["parsed_reports_df_file_name"],
        )
    )

    assert not extracted_reports.loc[report_id].empty

    report_text = extracted_reports.loc[report_id, "text"]
    headers = extracted_reports.loc[report_id, "headers"]

    assert report_text is not None

    extractor = ReportExtractor(report_text, report_id, headers)

    content_section, _ = extractor.extract_table_of_contents()

    print(f"Expected: {expected}")
    print(f"Actual: {content_section}")
    if expected:
        # Because we are now using a LLM to clean up the content section. We cant do an exact match. Instead we need to be atleast 2 out of 3 matches.
        assert (
            SequenceMatcher(
                None, content_section[: len(expected[0])].lower(), expected[0].lower()
            ).ratio()
            > 0.7
        )
        assert (
            SequenceMatcher(
                None, content_section[-len(expected[1]) :].lower(), expected[1].lower()
            ).ratio()
            > 0.7
        )
        assert abs(len(content_section) - expected[2]) < (expected[2] * 0.1)

    else:
        assert content_section is None


@pytest.mark.parametrize(
    "report_id, expected",
    [
        pytest.param(
            "TAIC_r_2019_102",
            [(1, 2), (7, 13), (13, 14), (14, 15)],
            id="TAIC_m_2019_102",
        ),
        pytest.param(
            "ATSB_a_2017_117",
            [(0, 1), (3, 5)],
            id="ATSB_a_2017_117 reading pdf headers",
        ),
        pytest.param("ATSB_a_2014_073", None, id="ATSB_a_2014_073 noy enough"),
        pytest.param(
            "TSB_m_2002_C0018",
            None,
            id="TSB_m_2002_C0018 random pdf headers not a content section",
        ),
        pytest.param(
            "TSB_a_2005_C0187",
            [(19, 23), (23, 25)],
            id="TSB_a_2005_C0187 relevant sections with different names",
        ),
        pytest.param(
            "ATSB_m_2017_003",
            [(10, 16), (17, 18)],
            id="ATSB_m_2017_003 needing to fill in the missing pages",
        ),
        pytest.param(
            "ATSB_a_2021_018",
            [("i", 1), (15, 17), (17, 18), (18, 21)],
            id="ATSB_a_2021_018 getting the extra safety issues section",
        ),
        pytest.param(
            "TSB_a_2020_P0013",
            [(31, 36), (36, 38)],
            id="TSB_a_2020_P0013 getting long content section",
        ),
        # pytest.param("TSB_a_2004_H0001", [29,30,31,32,33,34,35,36,37], id="TSB_a_2004_H0001 really messy content section") removed from testing as was too hard to read and get it too work. Extra long import
    ],
)
def test_safety_issue_content_section_reading(report_id, expected):
    extracted_reports = pd.read_pickle(
        os.path.join(
            pytest.output_config["folder_name"],
            pytest.output_config["parsed_reports_df_file_name"],
        )
    )

    assert not extracted_reports.loc[report_id].empty

    report_text = extracted_reports.loc[report_id, "text"]
    headers = extracted_reports.loc[report_id, "headers"]

    assert report_text is not None
    extractor = ReportExtractor(report_text, report_id, headers)
    content_section, _ = extractor.extract_table_of_contents()

    extractor = SafetyIssueExtractor(
        report_text, report_id, content_section, "full", "NA"
    )

    pages_to_read = extractor.extract_pages_to_read(content_section)
    print(f"Expected {expected} and got {pages_to_read}")
    print(f"Reading {content_section}")

    if expected is None:
        assert pages_to_read is None
    else:
        assert set(pages_to_read.to_tuples()).issuperset(expected)
        assert len(pages_to_read.pages) <= math.ceil(len(expected) * 1.6)


class TestSafetyIssueExtraction:
    def test_basic_colon(self):
        report_text = """
3.18. Regardless of w hy the system did not activate automatically, there were manual push -
button alarms located on every deck  that would have bypassed the automation and 
activated the fire alarm throughout the ship.  Pushing one of these buttons would have 
been the best method  of raising the alarm for everyone on board.  None of the crew 
pushed a manual button.  
Firefighting procedures and equipment  
Safety issue: Some aspects of the crew response to the fire did not follow industry good practice.  
3.19. Containing and extinguishing a fire quickly and effectively is critical for preserving life 
and property.  Once the initial individual direct attacks on the fire prove unsuccessful, the 
response would normally focus on containing the fire and using all ava ilable resources to 
extinguish it.  
3.20. The crew of the Dong Won 701  had all received approved shore -based firefighting 
training.  They were all required to be current with the vessel's firefighting procedures.  
This would normally have been achieved during th e crew induction on board and fire 
and emergency drills, which according to the operator's (DW New Zealand Limited 's) 
maritime transport operator plan14 were scheduled to happen four times each year.  
"""
        safety_issues = SafetyIssueExtractor(
            report_text, "TAIC_a_2020_001", report_text, "full", "TAIC"
        )._extract_safety_issues_with_inference(report_text)
        assert len(safety_issues) == 1
        assert (
            safety_issues[0]["safety_issue"]
            == "Some aspects of the crew response to the fire did not follow industry good practice."
        )

    def test_basic_multi_colon(self):
        report_text = """
3.18. Regardless of w hy the system did not activate automatically, there were manual push -
button alarms located on every deck  that would have bypassed the automation and 
activated the fire alarm throughout the ship.  Pushing one of these buttons would have 
been the best method  of raising the alarm for everyone on board.  None of the crew 
pushed a manual button.  
Firefighting procedures and equipment  
Safety issue: Some aspects of the crew response to the fire did not follow industry good practice.  
3.19. Containing and extinguishing a fire quickly and effectively is critical for preserving life 
and property.  Once the initial individual direct attacks on the fire prove unsuccessful, the 
response would normally focus on containing the fire and using all ava ilable resources to 
extinguish it.  
3.20. The crew of the Dong Won 701  had all received approved shore -based firefighting 
training.  They were all required to be current with the vessel's firefighting procedures.  
to do so by FENZ, and the master was unable to confirm that all his crew were safely off 
the vessel. Emergencies that occur in port are problematic in that regard, because not all 
crew will necessarily be on board.  In this case only nine of the 44 crew members were on 
board to initially contain and fight the fire.  Several important procedures were not 
followed, of which some were  likely attributable to so few crew being available to 
respond.  
3.22. An important aspect of containing a fire is to deprive it of oxygen.  The s tandard 
procedure for achieving this is to close all doors and openings to the space where the 
fire is located.  After the retreat from the initial attempts to make a direct  attack on the 
fire, the cabin door was left open – so too was the door leading from the passageway to 
the open deck.  Also, the ventilation flaps supplying fresh air to the accommodation were 
left open.  
3.23. With a free flow of oxygen to the fire, it quickly s pread from the first engineer’s cabin to 
engulf the accommodation spaces within a matter of minutes.  
3.24. The master’s designated position in an emergency was on the bridge.  However, by the 
time he was alerted to the fire the bridge was rapidly becoming engulf ed in fire.  
Consequently the master was unable to use the communication equipment to summon 
help.  
3.25. The master went to the open deck aft15 of the bridge , where a number of crew had 
mustered and attempted to organise a full muster and form teams to fight the fire.  
However, several crew  members took their own course s of action, with some proceeding 
to the engine room to manage the ammonia risk and others re -entering the 
accommodation space to continue attacking the fire on an individual basis.  
3.26. This unco ordinat ed response to the fire continued until FENZ arrived and took command 
of the scene, having been called by a member of the public.  
3.27. Some crew members attempted to initiate firefighting by preparing fire hoses on the 
port and starboard side s of the upper deck .  However, they were unable to work the 
portable emergency fire pump to pressurise the fire hoses.  
3.28. In summary, the delay in all crew being alerted to the fire was a missed opportunity to 
extinguish it using portable firefighting equipment before it escala ted out of control.  
Then, because the immediate and surrounding areas were not shut down, the fire was 
able to spread rapidly throughout the accommodation space before an effective 
Safety issue: Inconsistencies in the application of Rule 40D may have resulted in up to 12 fishing 
vessels operating under the New Zealand Flag not complying fully with the relevant safety 
                                                        
15 At, near or towards the stern of a ship.  
Final Report MO -2018 -202 
<< Page 13 >>
 standards. A further 50 fishing vessels have been afforded grandparent rights that will allow them 
to operate indefinitely without meeting contemporary safety standards under the current 
Maritime Rules.  
3.29. Fishing ships entered into the Fishing Vessel Register under the Fisheries Act 1996 (the 
Fisheries Act) are required  to meet applicable design , construction and equipment rules 
"""
        safety_issues = SafetyIssueExtractor(
            report_text, "TAIC_a_2020_001", report_text, "full", "TAIC"
        )._extract_safety_issues_with_inference(report_text)
        assert len(safety_issues) == 2
        assert [safety_issue["safety_issue"] for safety_issue in safety_issues] == [
            "Some aspects of the crew response to the fire did not follow industry good practice.",
            "Inconsistencies in the application of Rule 40D may have resulted in up to 12 fishing vessels operating under the New Zealand Flag not complying fully with the relevant safety standards. A further 50 fishing vessels have been afforded grandparent rights that will allow them to operate indefinitely without meeting contemporary safety standards under the current Maritime Rules.",
        ]

    def test_basic_hypen(self):
        report_text = """
4.3.5.  The servic ing staff and the train driver were qualified for their roles and all had experience in 
fitting and checking electrical connections bet ween locomotives .  KiwiRail reviewed its 
locomotive servicing procedures and used this derailment to highlight the need for train 
drivers to include visual examination s of electrical connection s when dealing with  any similar 
locomotive performance issues .  Consequently, t he Commission has not made any 
recommendations to KiwiRail to address this issue.  
4.4. Incorrect brake handle  set-up 
Safety issue - Driver B was able to set the brake handles incorrectly because there was no 
inter lock6 capability between the two driving cabs of the DL-class locomotive s.  The incorrect 
brake set -up resulted in  driver B not having brake control over the coupled wagons . 
4.4.1.  The DL-class  locomotive was  the first double -cab diesel locomotive to operate in New Zealand 
that had brake handles permanently fitted to both cabs.   It required train drivers to set the  
locomotive  and train brake handles correctly before vacating a cab and relocating to the cab 
at the other end.    
"""
        safety_issues = SafetyIssueExtractor(
            report_text, "TAIC_a_2020_001", report_text, "full", "TAIC"
        )._extract_safety_issues_with_inference(report_text)
        assert len(safety_issues) == 1
        assert (
            safety_issues[0]["safety_issue"]
            == "Driver B was able to set the brake handles incorrectly because there was no interlock capability between the two driving cabs of the DL-class locomotives. The incorrect brake set-up resulted in driver B not having brake control over the coupled wagons."
        )

    def test_basic_multi_hypen(self):
        report_text = """
4.3.5.  The servic ing staff and the train driver were qualified for their roles and all had experience in 
fitting and checking electrical connections bet ween locomotives .  KiwiRail reviewed its 
locomotive servicing procedures and used this derailment to highlight the need for train 
drivers to include visual examination s of electrical connection s when dealing with  any similar 
locomotive performance issues .  Consequently, t he Commission has not made any 
recommendations to KiwiRail to address this issue.  
4.4. Incorrect brake handle  set-up 
Safety issue - Driver B was able to set the brake handles incorrectly because there was no 
inter lock6 capability between the two driving cabs of the DL-class locomotive s.  The incorrect 
brake set -up resulted in  driver B not having brake control over the coupled wagons . 
4.4.1.  The DL-class  locomotive was  the first double -cab diesel locomotive to operate in New Zealand 
that had brake handles permanently fitted to both cabs.   It required train drivers to set the  
locomotive  and train brake handles correctly before vacating a cab and relocating to the cab 
at the other end.  
4.4.2.  In comparison , the e arlier -generation double -cab diesel  and electric locomotives , and the 
current fleet of double -cab EF -class electric locomotives , required some  or all of the brake 
handles  to be physically transferred by the drivers when changing ends.   This meant that train 
drivers had to isolate the brakes in a cab being vacated in order to remove the appropriate 
handles .  The handles were then carried to the other cab , where th ey were inserted to allow 
                                                        
5 An ammeter is a dial fitted on a locomoti ve's dashboard to display the amount of current being produced by 
the electrical generator.  
6 A mechanical or electrical feature that ensures that the operation of two linked mechanisms, brake handles 
in this instance, cannot be independently operated at t he same time . 
 

<< Page 12 >>
 Final Repot RO -2013 -101 the brakes  to be controlled from that end of the locomotive .  This manual process provided a  
reliable  interlock , prevent ing an incorrect brake handle set -up as seen in this instance . 
4.4.3.  KiwiRail investigated several options before choosing the Tranzlog event recorder system to 
provide an interlock capability  to prevent a similar incident .  The Tranzlog system was chosen 
to perform  this interlock b ecause the positions of the brake handles were  already being 
monitored by the system.  This option was successfully t ested during 2014/2015 and rolled 
out on all the DL-class  locomotives.  In view of this work, t he Commission has not made a 
recommendation to KiwiRail to address this issue.  
4.4.4.  Additionally , KiwiRail has committed to provid ing a similar Tranzlog -based interlock capability  
on all its other mainline diesel locomotives .  This work will supplement the existing manual 
interlock and was underway at the time of the compilation of this report.  
Finding s 
1. The train departed Gl enbrook with only the locomotive brakes operating.  The 
brakes on all of the wagons (the train brake) were not working because the driver 
had omitted to configure the brake levers correctly when changing from one 
driving cab to the other.  
2. The driver was ab le to depart with a wrongly configured brake system because 
there was no mechanical interlock to prevent his doing so.  
3. The required pre -departure train brake test would have revealed that the train 
brake was not operational, but those involved omitted to p erform the test when a 
third locomotive was attached to the train.  
4. The train parted when a coupling that had not been fully secured allowed the 
hook to spring up when the couplings suddenly compressed as the train 
descended the gradient from Glenbrook . 
5. The derailment was caused by the rear portion of the train colliding with the 
forward portion seconds after the train parted.  
4.5. Non-technical skills  
Safety issue - When the three staff members came together to couple the third  locomotive to 
the disabled train at Glenbrook, no challenge and confirm action s were taken to complete a 
fundamental brake test procedure , which was designed to ensure  that the trains ' air brakes 
were functioning correctly.  
4.5.1.  Non-technical skills (previously known as crew resource managemen t) are a set of practices 
designed to create a safe working environment, encourage teamwork, improve situational 
awareness and understand technical proficiency . 
4.5.2.  Staff using non-technical skills will communicate more effectively , be more aware of their 
situation, use all of its available resources  and work better with one another.   Communication 
skills and practices form a significant component of what has become known as non -technical 
skills in other transport modes.
"""
        safety_issues = SafetyIssueExtractor(
            report_text, "TAIC_a_2020_001", report_text, "full", "TAIC"
        )._extract_safety_issues_with_inference(report_text)
        assert len(safety_issues) == 2
        assert [s["safety_issue"] for s in safety_issues] == [
            "Driver B was able to set the brake handles incorrectly because there was no interlock capability between the two driving cabs of the DL-class locomotives. The incorrect brake set-up resulted in driver B not having brake control over the coupled wagons.",
            "When the three staff members came together to couple the third locomotive to the disabled train at Glenbrook, no challenge and confirm actions were taken to complete a fundamental brake test procedure, which was designed to ensure that the trains' air brakes were functioning correctly.",
        ]

    def test_complex_colon(self):
        report_text = """
24 A condition that occurs in vehicles when the angle of sunligh t hitting a windscreen creates glare that is very 
hard for a driver to see through . 
 

<< Page 14 >>
 Final Report RO-2020 -101 Mulcocks Road LCSIA  report and action taken  
Safety issue: SFAIRP assessments were not being routinely carried out for risk treatments 
recommended in LCSIA reports . No process , and minimal guidance , on SFAIRP asse ssment for 
level crossing risk treatments was available  in industry documents.  
Crossing closure  
 The LCSIA report's recommended 'Criterion  1' risk treatment , closure  of Mulcocks Road 
level crossing, would virtually certain ly have prevented this accident . 
 The LCSIA process requires 'Criterion 1'  risk treatments be given first considera tion. Its 
guidance document ation  provides  a flowchart , an excerpt of which is  shown in Figure 8 
below . The path overlaid in green  show s the process  that crossing closure should have 
followed,  arriving at 'Is treatment  suitable given constraints? ' This step represents a 
joint SFAIRP assessment  between KiwiRail and the Council . 
"""
        safety_issues = SafetyIssueExtractor(
            report_text, "TAIC_a_2020_001", report_text, "full", "TAIC"
        )._extract_safety_issues_with_inference(report_text)
        assert len(safety_issues) == 1
        assert (
            safety_issues[0]["safety_issue"]
            == "SFAIRP assessments were not being routinely carried out for risk treatments recommended in LCSIA reports. No process, and minimal guidance, on SFAIRP assessment for level crossing risk treatments was available in industry documents."
        )


class TestSectionExtraction:
    def __load_report_text(self, report_id):
        extracted_reports = pd.read_pickle(
            os.path.join(
                pytest.output_config["folder_name"],
                pytest.output_config["parsed_reports_df_file_name"],
            )
        )

        return extracted_reports.loc[report_id]["text"]

    def __test_section_extraction(self, report_id, section):
        report_text = self.__load_report_text(report_id)
        section = ReportSectionExtractor(
            report_text, "test report", None
        ).extract_section(section, useLLM=False)
        return section

    def test_section(self):
        report_text = """

<< Page 2 >>
 Final Report AO -2014 -005 1.12.  The Commission made t hree  recommendations  to the Director of Civil Aviation to address the 
safety issues.  
1.13.  The key lessons  arising from this inquiry  are: 
● flying in mountainous terrain places additional demands on a pilot's skills and an 
aircraft's performance.  Both could be at or near the limits of their capabilities.  Operators 
need to ensure that their safety management systems address the additional risks 
associated with flying in such an environment  
● the use of 'standard' or 'assessed' passenger weights is not a licence to exceed an 
aircraft's permissible weight and balance parameters.  Any aircraft being operated 
outside the permissible range will have a higher risk of having an accident, particularly 
when being operated near the margins of aircraft performance capability  
● it is important for operators to keep comprehensive, formal records of all pilot training.  
Historical training records provide the basi s for ongoing performance monitoring  and 
professional development , particularly given natural attrition as safety and training 
managers move through the industry  
● seatbelts are only effective in preventing or minimising injury if they are fastened and 
prope rly adjusted.  Aircraft operators must ensure that passengers and crew fasten their 
seatbelts and adjust them to fit tightly across their hips  
● vortex ring state is a known hazard for helicopters.  To avoid the hazard, pilots must:  
a.  remain alert to the c onditions conducive to the formation of vortex ring state  
b.  closely monitor the airspeed and rate of descent during the final approach  
c.  initiate recovery action at the first indication that they may be approaching vortex 
ring state.  
 
  
 
Final Report AO -2014 -005 
<< Page 3 >>
 2. Conduct of the inquiry  
2.1. On the afternoon of Saturday 16 August 2014, the Civil Aviation Authority (CAA) notified the 
Transport Accident Investigation Commission  (the Commission)  of the accident.  The 
Commission opened an inquiry under section 13(1)b of the Transport Accid ent Investigation 
Commission Act 1990, and appointed an investigator in charge.  
2.2. On 16 August 2014 the Commission notified the Bureau d'Enquêtes et d'Analyses (BEA) of 
France, which was the State of Manufacture for the helicopter and the engine.  In accorda nce 
with Annex 13 to the Convention on International Civil Aviation, France appointed a BEA 
investigator as its Accredited Representative to participate in the investigation.  
2.3. Three of the Commission's  investigators arrived in Queenstown on the afternoon o f 17 August 
2014 and conducted an initial survey of the accident site from a helicopter.  The investigators 
conducted a full site examination on 18 August 2014.  The wreckage was removed later that 
day and transported to the Commission's technical facility  in Wellington for further detailed 
examination.  
2.4. In the following three days, witnesses and first responders to the accident were interviewed.  
The maintenance records for the helicopter were obtained by the Commission and relevant 
engineering personnel we re interviewed.  The helicopter was not fitted with any equipment to 
record data, and no other source of recorded data was obtained from the accident flight.  
2.5. On 25 and 26 August 2014 the investigator in charge interviewed the surviving passengers.  
The inv estigation reviewed the CAA files concerning The Helicopter Line (the operator) and the 
pilot.  On 3 September 2014 the operator's general manager and chief pilot were interviewed.  
2.6. On 14 November 2014, at the request of the Commission, the helicopter manu facturer, Airbus 
Helicopters, completed an analysis of relevant helicopter performance data.  
2.7. On 28 January 2015, once the snow had melted, a team searched the accident site for any 
unrecovered items.  The search team recovered some items from the helicopte r. 
2.8. Between December 2014 and February 2015 , the Commission obtained additional 
information through BEA concerning the helicopter seats and seatbelts.  
2.9. On 19 October 2015 the engine was examined by Turbomeca at its maintenance facility in 
Sydney, Australia.   The Australian Transport Safety Bureau appointed an Accredited 
Representative to the Commission's inquiry to supervise the examination on behalf of the 
Commission.  On 29 October 2015 the Australian Transport Safety Bureau provided the 
Commission with a report on the engine examination.  
2.10.  On 27 January 2016 three investigators from BEA and a senior investigator from Airbus 
Helicopters travelled to New Zealand and examined the wreckage of the helicopter at the 
Commission's technical facility.  
2.11.  On 24 August 20 16 the Commission approved the circulation of the draft report to interested 
persons for comment  and received submissions from four interested persons.   
2.12.  In response to the submissions received, the Commission undertook further independent 
enquiries , which included a review of all the primary evidence and information it had received 
from all sources.  The Commission also sought the opinion of an expert who had a long 
association with heli -skiing operations in Canada as a helicopter pilot, a regulator  and an 
independent accident investigator.  
2.13.  On 24 May 2017  the Commission approved the circulation of a revised draft report to 
interested persons affected by the changes .  Substantive submissions were received from 
three interested persons , including the o perator and the regulator  (CAA) .  The Commission 
requested further information concerning those submissions . 

<< Page 4 >>
 Final Report AO -2014 -005 2.14.  On 23 August 2017  the Commission approved the circulation of a 2nd revised  draft report to 
interested persons for their  comment.   Further submissions were received and these were 
considered in the preparation of the final report.  
2.15.  On 25 October 2017, the Commission deferred publication of its final report due to 
prosecution proceedings the CAA and the operator were parties to, and w hich was proceeding 
to trial in November. The Commission wanted to ensure its report did not affect the fair 
administration of justice.  
2.16.  On 16 November 2017, the Commission approved publication of its final report following 
notification the prosecution pro ceedings were not proceeding to trial.  
  
 
Final Report AO -2014 -005 
<< Page 5 >>
 3. Factual information  
3.1. Narrative  
3.1.1.  On the evening of Friday 15 August 2014, the pilot and the heli -skiing co -ordinator for the 
operator discussed the plan for the next day.  There were three skiing groups3 with whom the 
pilot would be flying the next day.  
3.1.2.  The heli -skiing was initially organised through an associated company (the heli -ski provider), 
which had taken the bookings and organised guides for each group.4  Five of the operator's 
helicopters were to  be used to support heli -skiing groups during the day, mainly in the 
mountains between Mount Aspiring National Park and Lake Wanaka.  
16 August 2014  
3.1.3.  At 0754 on Saturday 16 August 2014, the pilot took off from Queenstown in an AS350 -B2 
helicopter and took a group of passengers to the Cardrona snow park. He then continued to 
Wanaka where he delivered the helicopter to the maintenance provider.  The pilot then picked 
up another AS350 -B2 helicopter, the one involved in the accident (the helicopter).  The 
operato r had recently imported the helicopter from the United States, and this was to be its 
first commercial flight in New Zealand.  
3.1.4.  The pilot met with the engineer who had supervised the maintenance check and fitting of 
equipment to the helicopter.  Together the y transferred the ski basket  and other equipment, 
checked the documentation and conducted a visual inspection of the helicopter.  The pilot 
then flew the helicopter back to Queenstown to meet up with the ski groups.  
3.1.5.  At 1002 the pilot landed back at Queenst own and the three groups were  given their helicopter 
safety briefing in preparation for the day's activities.  A heli -ski guide loaded  the first group of 
five, group A, and their gear onto the h
"""
        section = ReportSectionExtractor(report_text, 1).extract_section(
            "2", useLLM=False
        )

        assert (
            section
            == """2.1. On the afternoon of Saturday 16 August 2014, the Civil Aviation Authority (CAA) notified the 
Transport Accident Investigation Commission  (the Commission)  of the accident.  The 
Commission opened an inquiry under section 13(1)b of the Transport Accid ent Investigation 
Commission Act 1990, and appointed an investigator in charge.  
2.2. On 16 August 2014 the Commission notified the Bureau d'Enquêtes et d'Analyses (BEA) of 
France, which was the State of Manufacture for the helicopter and the engine.  In accorda nce 
with Annex 13 to the Convention on International Civil Aviation, France appointed a BEA 
investigator as its Accredited Representative to participate in the investigation.  
2.3. Three of the Commission's  investigators arrived in Queenstown on the afternoon o f 17 August 
2014 and conducted an initial survey of the accident site from a helicopter.  The investigators 
conducted a full site examination on 18 August 2014.  The wreckage was removed later that 
day and transported to the Commission's technical facility  in Wellington for further detailed 
examination.  
2.4. In the following three days, witnesses and first responders to the accident were interviewed.  
The maintenance records for the helicopter were obtained by the Commission and relevant 
engineering personnel we re interviewed.  The helicopter was not fitted with any equipment to 
record data, and no other source of recorded data was obtained from the accident flight.  
2.5. On 25 and 26 August 2014 the investigator in charge interviewed the surviving passengers.  
The inv estigation reviewed the CAA files concerning The Helicopter Line (the operator) and the 
pilot.  On 3 September 2014 the operator's general manager and chief pilot were interviewed.  
2.6. On 14 November 2014, at the request of the Commission, the helicopter manu facturer, Airbus 
Helicopters, completed an analysis of relevant helicopter performance data.  
2.7. On 28 January 2015, once the snow had melted, a team searched the accident site for any 
unrecovered items.  The search team recovered some items from the helicopte r. 
2.8. Between December 2014 and February 2015 , the Commission obtained additional 
information through BEA concerning the helicopter seats and seatbelts.  
2.9. On 19 October 2015 the engine was examined by Turbomeca at its maintenance facility in 
Sydney, Australia.   The Australian Transport Safety Bureau appointed an Accredited 
Representative to the Commission's inquiry to supervise the examination on behalf of the 
Commission.  On 29 October 2015 the Australian Transport Safety Bureau provided the 
Commission with a report on the engine examination.  
2.10.  On 27 January 2016 three investigators from BEA and a senior investigator from Airbus 
Helicopters travelled to New Zealand and examined the wreckage of the helicopter at the 
Commission's technical facility.  
2.11.  On 24 August 20 16 the Commission approved the circulation of the draft report to interested 
persons for comment  and received submissions from four interested persons.   
2.12.  In response to the submissions received, the Commission undertook further independent 
enquiries , which included a review of all the primary evidence and information it had received 
from all sources.  The Commission also sought the opinion of an expert who had a long 
association with heli -skiing operations in Canada as a helicopter pilot, a regulator  and an 
independent accident investigator.  
2.13.  On 24 May 2017  the Commission approved the circulation of a revised draft report to 
interested persons affected by the changes .  Substantive submissions were received from 
three interested persons , including the o perator and the regulator  (CAA) .  The Commission 
requested further information concerning those submissions . 

<< Page 4 >>
 Final Report AO -2014 -005 2.14.  On 23 August 2017  the Commission approved the circulation of a 2nd revised  draft report to 
interested persons for their  comment.   Further submissions were received and these were 
considered in the preparation of the final report.  
2.15.  On 25 October 2017, the Commission deferred publication of its final report due to 
prosecution proceedings the CAA and the operator were parties to, and w hich was proceeding 
to trial in November. The Commission wanted to ensure its report did not affect the fair 
administration of justice.  
2.16.  On 16 November 2017, the Commission approved publication of its final report following 
notification the prosecution pro ceedings were not proceeding to trial.  
  
 
Final Report AO -2014 -005 
<< Page 5 >>
 3. Factual information"""
        )

    def test_subsection(self):
        report_text = """
<< Page 1 >>
 1. Executive summary  
1.1. The Liberian -registered container ship Rena  had left the New Zealand port of Napier at 1020 
on 4 October 2011 and was bound for the New Zealand port of Tauranga.  The master had 
given an estimated time of arrival at the Tauranga pilot station of 0300 the next day. The 
master calculated the estimated time of arrival by dividing the distance to go by the Rena 's 
normal service speed.  The calculation did not account for the unfavourable  currents that 
normally prevail ed down that stretch of coastline.  
1.2. After departure from Napier the master  learned  from notes on the chart of the unfavourable  
currents.  He then authorised the watchkeepers to deviate from the planned course lines on 
the chart to shorten the distance , and to search for the least unfavourable curr ents.  
1.3. The Rena 's second mate took over the watch shortly after midnight on 4 October.  He 
calculated that the Rena  would arrive at the port of Tauranga pilot station at 0300  at the 
ship's  then current speed.  Times for ships entering and leaving Tauranga H arbour are limited 
by the depth of water and the strength of the tidal currents  in the entrance channel .  Tauranga 
Harbour Control informed the second  mate that the latest time the Rena  could take the 
harbour pilot on board was 0300 .   
1.4. The planned course t o the Tauranga pilot station was to pass two nautical miles north of 
Astrolabe Reef before making the final adjustment in course to the pilot station.   The second  
mate decided to reduce the two miles to one mile in order to save time.  The second  mate 
then made a series of small course adjustments towards Astrolabe Reef to make the shortcut . 
In doing so he altered the course 5 degrees past the required track and did not make an 
allowance for any compass error or sideways "drift", and as a conseque nce the Rena  was 
making a ground track directly for Astrolabe Reef.  Meanwhile the master had been woken 
and arrived on the bridge to prepare for arrival at the port.  
1.5. The master and second  mate discussed preparations for arrival at the pilot station.  The 
master then assumed control of the ship , having received virtually no information on where 
the ship was , where it was heading , and what immediate dangers to navigation he needed to 
consider.  
"""
        section = ReportSectionExtractor(report_text, "test report").extract_section(
            "1.4", useLLM=False
        )

        assert section is not None
        assert (
            section
            == """1.4. The planned course t o the Tauranga pilot station was to pass two nautical miles north of 
Astrolabe Reef before making the final adjustment in course to the pilot station.   The second  
mate decided to reduce the two miles to one mile in order to save time.  The second  mate 
then made a series of small course adjustments towards Astrolabe Reef to make the shortcut . 
In doing so he altered the course 5 degrees past the required track and did not make an 
allowance for any compass error or sideways "drift", and as a conseque nce the Rena  was 
making a ground track directly for Astrolabe Reef.  Meanwhile the master had been woken 
and arrived on the bridge to prepare for arrival at the port."""
        )

    def test_section_full_report(self):
        section = self.__test_section_extraction("TAIC_m_2016_204", "4.3")

        assert (
            section
            == """4.3. Passage planning  
Safety issue: The vessel 's bridge team and the pilot did not have a shared understanding of  a 
common passage plan before the pilotage began.  Consequently,  the pilot and the vessel 's 
bridge team had different understanding s of the planned track to be followed and their 
respective roles in monitoring against the plan.  
Safety issue : The IMO has set standards for passage planning that vessel s must adhere to, 
but there is no corresponding requirement for the passage plans that port authorities  create 
and use to meet those same standards . 
4.3.1.  When a pilot joins a vessel prior to the pilotage it is the first opportunity for the master and 
bridge team to : talk to the pilot; clarify any issues that may have been identified during the 
preparation of the passage plan; and ensure they are satisfied with the planned transit.  At 
this stage the pilot needs to be fully integrated into the bridge organisation so that the whole 
team works as a cohesive body and has a shared understanding  of the p assage plan . 
4.3.2.  Before any pilotage act begins, it is essential that the pilot and the master (including other 
members of the team) have a shared understanding of  the passage plan.  That did not happen 
in this case.  As often happens, the pilot intended  to na vigate to the standard port company 
passage plan , and the vessel  had its own passage plan, which  differed  from that of the port 
company . 
4.3.3.  An essential part of integrating the pilot into the bridge team is the pilot and master exchange 
of information, a briefing that should include all members of the bridge team.  The briefing is a 
bilateral exchange of important information where everyone is made aware of : any changes to 
the proposed plan ; the handling characteristics of the vessel ; and any notable 'dyna mic' 
information such as weather and tides  for the transit.   
4.3.4.  The port passage plan was available over the internet from the Port Otago  website .  The 
channel is narrow and there is little scope for deviating from the plan without leaving the 
navigational c hannel.  The preferred courses were presented as  smooth , curved lines without 
any marked waypoints, turn radii or off -track limits.  Without that information the vessel 's crew 
would not have been  able to replicate it in their own navigation systems, such a s the ECDIS .  
The vessel  was required to plan the passage in accordance with the IMO convention standards 
and guidelines16 (see Appendix 3) , but there was no international or New Zealand requirement 
for the port companies and their pilots to follow the same  standards when developing their 
own generic passage plans.  The Port Otago passage plan as presented on its website would 
not meet the IMO  standards  or other reputable guidelines available to mariners on voyage 
planning17.   
4.3.5.  When the pilot joined the Molly  Manx  he and the master discussed the vessel's characteristics 
and went through the Otago Harbour passage  plan.  
4.3.6.  However, the vessel  had its own passage plan loaded into the ECDIS, which differed from that 
of Port Otago.  Because the vessel 's plan was the p lan loaded into the ECDIS, that is what the 
bridge team , excluding the pilot, was using to monitor the vessel's progress.  Me anwhile , the 
pilot was navigating the vessel  to the Port Otago  passage plan, using visual reference s as he 
had been  trained to do .  The bridge team, which now included the pilot, was not aligned in its 
thinking and did not share the same understanding  of the plan.  This dynamic was going to 
make effective BRM  difficult to achieve, which is discussed in the following section.  
4.3.7.  The New  Zealand Port and Harbour Marine Safety Code  is a voluntary national standard .  The 
Code recommends that , "Up-to-date passage plans and guidance should be published, and be 
available to harbour users and the masters of visiting vessel s" (see Appendix 6).  
                                                        
16 Chapt er V, Safety of Navigation, of the Annex to the International Convention for the Safety of Life at Sea 
and Resolution A.893(21) Guidelines for Voyage Planning.  
17 Other best -practice guidelines also contain valuable advice on bridge watchkeeping in general and voyage 
planning in particular. They include: the United Kingdom's Maritime and Coastguard Agency's guidance on 
Chapter V, Safety of Navigation, of the Annex to the International Convention for the Safety of Life at Sea ; the 
Nautical Institute's Bridge Team Management - A practical guide; and the International Chamber of 
Shipping's Bridge Procedures Guide.  

<< Page 15 >>
 
Final Report MO -2016 -204 | Page 15 4.3.8.  One method of ensuring that an approved passage plan is available on board would be for 
port companies or harbour authorities to make available to vessel s properly constructed and 
validated passage plan s that meet the  port-specific standards  and guidelines included in  
Chapter V, Safety of Navigation , of the Annex to the International Convention for the Safety of 
Life at Sea  (SOLAS) , and Resolution A.893(21) Guidelines for Voyage Planning .  Such a system 
would assist in on-board passage planning and allow a v essel to be better prepared when the 
pilot boards .  This action would greatly assist the smooth transition of the pilot into the bridge 
team at a time of typically high workload and little time before the pilotage begins.  
4.3.9.  More vessel s are using ECDIS s as the primary means of navigation , and  this will increase in 
the future.  As it was  on board the Molly Manx , the passage plan to the berth is usually loaded 
into the vessel 's ECDIS.  Ideally, passage plans generated by port companies should be to the 
same IMO standard s that vessel s are required to meet , and should be compatible for use in 
an ECDIS.   
4.3.10.  Many vessel s transit more than one New Zealand port.  It would greatly enhance safety if the 
passage plans were, as far as practicable, in a standard format an d could be found at one 
site.  Vessel s routed to several New Zealand ports would be able to access from one place 
standardised passage plans for several ports, even before they depart ed from their previous 
overseas port s. 
4.3.11.  Currently there can be issues with uploading standardised passage plans into an ECDIS, 
because ECDIS manufacturers have proprietary systems that require specific formats.  
However,  that will shortly change.   The International Hydrographic Organization  and the 
International Electrotechnical  Commission standard for ECDIS s (IEC 61774 Edition 4, 
September 2015) from August 2017 includes a route exchange format that will make  it easier 
for data transfers.  In the future it will be possible to send passage plans to all vessels in the 
correct form at to be uploaded direct ly into the ECDIS system , thereby reducing the possibility 
of navigating officers making errors when loading them  into ECDIS s.  However, this facility was 
not available at the time of the accident and therefore was not able to be us ed by  the bridge 
team on this occasion.    
4.3.12.  The Commission has made recommendations to Maritime New Zealand to promote the use of 
standard passage plans by all New Zealand harbour authorities."""
        )

    def test_paragraph(self):
        report_text = """
Final report 11 -204 
<< Page 23 >>
 1. Plot the new position for the course line  alteration point north of Astrolabe Reef  
2. Plot the vessel's  position  
3. Draw a new line to the new course line  alteration point  north of Astrolabe Reef  
4. Make the alteration o f gyro heading  and closely monitor  the s hip's progress  along 
that track .  The ship 's progress should have been monitored more closely than 
before  the deviation from the passage plan,  given the higher risk created  by the new 
track . 
4.2.24.  The second  mate did none of the above.  T he alteration to the gyro heading  was not a single 
change to the vessel 's heading but rather a progressive change over about 30 minutes.  The 
true course between the pin-prick  he made on the chart  and the new intended alteration point 
one nautical mile north of the reef was about 260 degrees.  
4.2.25.  The incremental alteration to the gyro heading  suggest s that the second mate had not fully 
considered what the new gyro heading  would be.  While altering the vessel 's heading (set by 
the automatic pilot) the second mate  went 5 degrees past the 260 -degree required track and 
he did not account for set, leeway or gyrocompass error.  The deviation was done in an ad -hoc 
manner and no consideration was given as to ho w the Rena 's progress would be measured 
against the amended plan.  Once the vessel's heading had stabilised on about 2 55 degrees at 
about 0150 the Rena  was heading for a collision with Astrolabe Reef, and the crew were not 
aware of the imminent danger.  
Findings:  
"""

        section = ReportSectionExtractor(report_text, "test report").extract_section(
            "4.2.24", useLLM=False
        )
        assert (
            section
            == """4.2.24.  The second  mate did none of the above.  T he alteration to the gyro heading  was not a single 
change to the vessel 's heading but rather a progressive change over about 30 minutes.  The 
true course between the pin-prick  he made on the chart  and the new intended alteration point 
one nautical mile north of the reef was about 260 degrees."""
        )

    def test_paragraph_to_next_sub_section(self):
        report_text = """
Final Report MO -2017 -203 
<< Page 17 >>
 4. Analysis  
4.1. Introduction  
4.1.1.  Pressure vessels are widely used on  board ships for various applications .  Their  failure can be 
catastrophic and may result  in human injury and death , as was the case on board  the Emerald 
Princess .  
4.1.2.  The crew were follow ing the correct procedure for recharging the nitrogen cylinders when the 
cylinder burst.  The failed cylinder was one of four located at lifeboat station No.  24 and had 
very little to no protection from sea spray.  The cylinder was severely weakened by corrosion , 
which caused it to fail under normal working loads .  The shipboard maintenance plan and the 
various inspection regimes that gave effect to that plan did not detect or remedy the issue 
before the failure occurred.  T he cylinder that failed was  overdue for maintenance  and testing,  
and should not have been in service at the time of the accident.  
4.1.3.  The following analysis discusses why the nitrogen cylinder  remain ed in service despite being in 
a danger ous condition .  The analysis raise s the following two safety issues : 
 there are currently no global minimum standards for the inspection, testing and rejection 
of pressure cylinders that make up part of stored e nergy systems on lifeboat launching 
installations, which has resulted in a wide variation in, and in some case inadequate, 
standards applied by flag state administrations, classification societies and authorised 
service providers  
 technicians who are author ised to conduct mandatory annual and five -yearly inspections 
of lifeboat -launching installations are not required to have specific training and 
certification for inspecting any stored energy -release systems and their associated 
pressure cylinders.  
4.2. Why did the cylinder burst?  
4.2.1.  The cylinder burst because a region of severe corrosion reduced the cylinder wall thickness 
from about 4.81  mm to 1.45  mm and significantly compromised the cylinder's ability to 
withstand internal pressure.  As a result, the cylinder su ffered an overpressure burst while the 
crew attempted to raise the nitrogen pressure from 160 bar to its normal working pressure of 
200 bar.  
4.2.2.  There is no clear evidence as to why this particular cylinder was so heavily corroded in the 
area where the failure  started.  The adjacent cylinders in the frame were not heavily corroded. 
Corrosion can be initiated by a number of factors.  Mechanical damage to the protective 
coating in a saltwater environment is a common cause.  Seawater being trapped against a 
steel surface can accelerate corrosion.  The straps that retained the bottles in the frame is a 
place where seawater could have become trapped.  They could also  have  initiate d mechanical 
damage to the protective coating.  However, as Figure 15 shows, the locatio n where the worst 
corrosion occurred was not where the straps were in contact with the bottles.  
4.2.3.  There was no evidence of any pre -existing crack occurring before the failure.  The source of 
the leak in the system that the crew were trying to remedy was neve r found.  
 
Final Report MO -2017 -203 
<< Page 18 >>
  
Figure 15 
Location of burst on failed cylinder  
4.3. On-board maintenance and inspection of pressure vessels  
Maintenance   
4.3.1.  The maintenance schedule for nitrogen cylinders and accumulators  was calendar based .  That 
is, time was the trigger for the various inspections and test s.  Weekly and monthly routines 
were carried out by the crew , while major maintenance work such as h ydrostatic pressure 
testing of nitrogen cylinders  was designated to an authorised service provider  or sh ore-based 
testing facility . 
4.3.2.  Hydrostatic pressure testing is a standard method of pressure testing cylinders.  It involves 
filling a cylinder with water and pressuris ing it up to 1.5 times its design pressure limit.  The 
pressure is then held for a prescribed amount of time and the cylinder is  then  inspected for 
leaks.  
4.3.3.  The maintenance regime on board the Emerald Princess  was kept in a  computer -based 
maintenance  system.  
4.3.4.  The hydrostatic pressure testing of nitrogen cyli nders was scheduled to have been carried out 
every 10 years , starting from the date they were  first h ydrostatically pressure tested. H owever, 
the start date entered into the maintenance plan was May 2007,  which was  the date that the 
nitrogen cylinders were installed on the vessel during b uild.  However, the  cylinder that failed  
had been  manufactured in September 2005, which was when it was first pressure tested.  
Therefore , it was overdue for testing by one  year and five  months at the time of the accident.  
The operator has now addressed this issue and updated its maintenance plan so that it tracks 
the age of each cylinder based on  its manufacture date , rather than its installation date . 

Final Report MO -2017 -203 
"""
        section = ReportSectionExtractor(report_text, "test report").extract_section(
            "4.2.3", useLLM=False
        )

        assert (
            section
            == """4.2.3.  There was no evidence of any pre -existing crack occurring before the failure.  The source of 
the leak in the system that the crew were trying to remedy was neve r found.  
 
Final Report MO -2017 -203 
<< Page 18 >>
  
Figure 15 
Location of burst on failed cylinder"""
        )

    def test_paragraph_to_next_section_full(self):
        section = self.__test_section_extraction("TAIC_a_2014_004", "4.2.5")

        assert (
            section
            == """4.2.5.  A toxicologic al examination of the pilot found no performance -impairing substances and the 
carbon monoxide level in his blood was in the normal range for a smoker.  It was therefore 
very unlikely that the pilot had been physically incapacitated before the stall.  
                                                        
29 CAA Medical Examiners' - Medical Manual Part 3 - Clinical Aviation Medicine  
30 Dr Rob Griffiths MB, ChB(Hons), FAFPHM, FAFOM, MMP, DIH , DipAvMed, FFOM(RCP), FACASM, FACOEM  Finding  
1. Pilot incapacitation was very unlikely to have been a contributing factor.  

<< Page 14 >>
Page 14 | Final report AO -2014 -004"""
        )

    def test_paragraph_to_next_section(self):
        report_text = """
25   
 
Final Report MO -2017 -203 
<< Page 16 >>
 the annual and five -yearly  inspections be authorised by the flag state and qualified to 
examine, test and repair each make and type of equipment for which they provide d service.   
The applicable s tandards and testing regime s for nitrogen cylinders and accumulators are 
discussed in detail in the a nalysis section.  
3.6.4.  Annual  and five-yearly inspections were carried out on board the Emerald Princess .  At the 
time of the accident the most recent inspection  had been  a five -yearly inspection  that was  
carried out on 21 January 2017,  19 days prior to the accident.   
3.6.5.  The five-yearly inspection  report  prepared by the authorised service provider14 stated that the 
nitrogen cylinders on  board the Emerald Princess  were aged and the company should 
consider swap ping them with new nitrogen cylinders.   The report also stated th at at least one  
accumulator was  corroded and the operator should  consider engaging the equipment 
manufacturer  to overh aul and certify the accumu lator .  (See Appendix  1 for relevant sections  
from the authorised service provider's  report .)  
                                                        
14 A person or company that has received approval to service or carry out work on a specified piece of 
equipment.  
Final Report MO -2017 -203 
<< Page 17 >>
 4. Analysis  
4.1. Introduction  
4.1.1.  Pressure vessels are widely used on  board ships for various applications .  Their  failure can be 
catastrophic and may result  in human injury and death , as was the case on board  the Emerald 
Princess .  
4.1.2.  The crew were follow ing the correct procedure for recharging the nitrogen cylinders when the 
cylinder burst.  The failed cylinder was one of four located at lifeboat station No.  24 and had 
very little to no protection from sea spray.  The cylinder was severely weakened by corrosion , 
which caused it to fail under normal working loads .  The shipboard maintenance plan and the 
various inspection regimes that gave effect to that plan did not detect or remedy the issue 
before the failure occurred.  T he cylinder that failed was  overdue for maintenance  and testing,  
and should not have been in service at the time of the accident.  
4.1.3.  The following analysis discusses why the nitrogen cylinder  remain ed in service despite being in 
a danger ous condition .  The analysis raise s the following two safety issues : 
 there are currently no global minimum standards for the inspection, testing and rejection 
of pressure cylinders that make up part of stored e nergy systems on lifeboat launching 
installations, which has resulted in a wide variation in, and in some case inadequate, 
standards applied by flag state administrations, classification societies and authorised 
service providers  
"""
        section = ReportSectionExtractor(report_text, "test report").extract_section(
            "3.6.5", useLLM=False
        )

        assert (
            section
            == """3.6.5.  The five-yearly inspection  report  prepared by the authorised service provider14 stated that the 
nitrogen cylinders on  board the Emerald Princess  were aged and the company should 
consider swap ping them with new nitrogen cylinders.   The report also stated th at at least one  
accumulator was  corroded and the operator should  consider engaging the equipment 
manufacturer  to overh aul and certify the accumu lator .  (See Appendix  1 for relevant sections  
from the authorised service provider's  report .)  
                                                        
14 A person or company that has received approval to service or carry out work on a specified piece of 
equipment.  
Final Report MO -2017 -203 
<< Page 17 >>
 4. Analysis"""
        )

    def test_missing_next_section(self):
        section = self.__test_section_extraction("TAIC_m_2010_204", "3.1.9")

        assert (
            section
            == """3.1.9.  At about 1957 the vessel was approximately abeam of No.4 buoy .  The pilot was unable to 
ascertain from the master what the problem was with the main  engine , so he radioed  the tugs 
to return and stand  by to assist as soon as possible.  
  

<< Page 6 >>
Page 6 | Report 10 -204"""
        )

    def test_suitable_match_after_real_section(self):
        section = self.__test_section_extraction("TAIC_m_2016_204", "4.2.10")

        assert (
            section
            == """4.2.10.  The pilot realised that  something was wrong when he , along with the rest of the bridge team , 
felt a bump .  He noticed that the speed of the vessel was slowing and the vessel's bow was 
swing ing to starboard despite  his havin g just applied port hel m.  He realised from these 
indicators that the vessel was grounding  and immediately ordered the engine  to stop and then 
astern and for the tug secured aft to pull right aft .  In doing so the  pilot was able  to manoeuvre 
the vessel off the bank and back into  deeper water . 
Finding s 
1. The grounding occurred because the bridge team , including the pilot , lost 
situational awareness and did not realise that the vessel  had deviated so far 
starboard of the intended track . 
2. The bridge team , including the pilot , did not realise how far the vessel  had 
deviated from the intended track because they were not monitoring the vessel 's 
progress effectively and by all available means . 

<< Page 13 >>
 
Final Report MO -2016 -204 | Page 13 Figure 5  
Passage plan track of the Molly Manx  (green) and actual track (red)  
  
passage plan track  
Molly Manx 's ship's head  
ebb tide direction  
ebb tide direction  
port authority cargo sheds  
ebb tide direction  
grounding position  
Molly Manx 's course made 
good  

<< Page 14 >>
Final Report MO -2016 -204 | Page 14"""
        )

    def test_skipped_section_mismatch(self):
        section = self.__test_section_extraction("TAIC_a_2010_001", "1.3")

        assert (
            section
            == """1.3. The investigation found that the events might have been avoided or been less severe ha d the 
operator had a more robust flight dispatch system, and had the air traffic service complied fully 
with a requirement to pass flight information to pilots on first contact.  The Commission made a 
safety recommendation regarding the clarity of informat ion about  hazardous meteorological 
conditions.  

<< Page 2 >>
 
Page 2 | Report 10 -001 2. Factual Information"""
        )

    def test_no_section(self):
        report_text = """

Final Report MO -2017 -203 | Page iv Data summary  
Vehicle particulars  
Name:  Emerald P rincess   
Type:  passenger ship  
Limits:  unlimited  
Classification:  Lloyds Register  
Length:  288.61 metres  
Breadth:  36.05 metres  
Draught:   8.6 metres  
Gross tonnage:  113,561  
Built:  2007   
Propulsion:  diesel electric (six Wärtsilä  engines ) 
Service speed:  22 knots  
Owner/operator:  Princess Cruise Lines L imited  
Port of registry:  Hamilton, Bermuda  
  
Date and time  9 February 2017 at 1700  
Location  at the berth in Port Chalmers, Dunedin  
Persons on board  
 
Injuries  passengers:  3,113  
crew:    1,173 
one fatality   
Final Report MO -2017 -203 
<< Page 1 >>
 1. Executive summary  
1.1. On 9 February 2017, the crew of the Bermuda -flagged passenger ship Emerald Princess  were 
conducting maintenance on one of the lifeboat launching systems while the ship was berthed 
at Port Chalmers, Dunedin.  
1.2. The maintenance was completed and the crew were restoring pressure to a b ank of high -
pressure nitrogen -gas cylinders that formed part of the launching davit 'stored energy' system . 
One of the nitrogen bottles burst, fatally injuring a crew member who was standing close by.  
1.3. The Transport Accident Investigation Commission (Commis sion) found  that the nitrogen 
cylinder  burst at below its normal working pressure because severe external corrosion had 
reduced the wall thickness to about 30% of its original thickness.  
1.4. The failed nitrogen cylinder and several other pressure cylinders within the stored energy 
system, despite having been surveyed about two weeks earlier , were not fit for purpose and 
should not have been left in service.  
1.5. The Commission also found  that there  is an urgent need for consistent and proper standards 
to be developed at a global level for maintaining, inspecting, testing and, where necessary, 
replacing high -pressure cylinders associated with stored energy systems on board ships.  
1.6. The operator took a number of immediate safety actions  to prevent  a recurrence of the 
accident  on any of its ships . 
1.7. The Commission issued an interim report with early  recommendations  to the equipment 
manufacturer , the International Soci ety of Classification Societies , the Cru ise Lines 
International Association and Maritime New Zealand to alert their members and surveyors as 
appropriate to the circumstances of the accident and to have the condition of similar 
installations checked.  
1.8. The Commission made two additional recommendat ions: one for the manufacturer to improve 
training for its surveyors; and one for Maritime New Zealand to raise , through the appropriate 
International Maritime Organi zation safety committee for its consideration, the implications for 
maritime safety of not  having adequate minimum standards for the inspection , testing and 
rejection of pressure vessels that are part of stored energy system s. 
1.9. A key lesson  arising from this inquiry is:  
 any sign of corrosion on high -pressure cylinders should be fully investigated by a person 
competent in examining high -pressure cylinders before any remedial work is undertaken 
and the cylinder s are  allowed back into service . 

"""

        section = ReportSectionExtractor(report_text, "test report").extract_section(
            "8.6", useLLM=False
        )
        assert section is None

    def test_no_section_prior_match(self):
        section = self.__test_section_extraction("TAIC_r_2022_101", "4.5")

        assert section is None

    def test_section_with_fallback_next_section(self):
        """
        There can be a situation where it will go from a sub paragraph to the next paragraph, as the subsection is not findable.
        """

        section = self.__test_section_extraction("TAIC_a_2010_009", "3.6.59")

        assert (
            section
            == """3.6.59.  Between 11 and 18 August 2010, in anticipation of the introduction of R ule Part 115 
Adventure Aviation  and to get an indication of the level of safety of parachute operations , the 
HSE Unit completed audit inspections o f4 of the larger parachuting operators  located in the 
central North Island.  Rule Part 115  was intended to cover commercial activities previously 
exempted, including parachute -drop and tandem -parachute operations.  The manager  
commented that he thought the standard of personal safety was good at the 4 operators they 
had inspected .  The manager had no piloting or engineering experience and was not able to 
comment on operational safety and compliance matters regarding the 4 operators.  Follow -up 
educational material was later passed to the NZPIA f or distribution to member organisations.  
 
  
                                                        
12 Supplement to New Zealand Gazette of 1 May 2003, dated 5 May 2003 - Issue No.44, Department of Labour Health and Safety in 
Employment Act 1992 Prime Ministerial Designation Pursuant to Section 28B of the Health and Safety in Employment Act 1992.  

<< Page 20 >>
Page 20 | Report 10 -009"""
        )

    def test_first_section(self):
        section = self.__test_section_extraction("TAIC_r_2019_106", "1.1")

        assert (
            section
            == """1.1 At about 1802  on 3 September 2019 , the daily TranzAlpine service was approaching 
Rolleston station on a return journey from Greymouth to Christchurch. The train was 
incorrectly routed onto the West Main Line. The platform at Rolleston could not be 
accessed from this track , and two passenge rs who disembarked walked across the East 
Main Line without the knowledge of train control ."""
        )

    def test_two_early_references(self):
        section = self.__test_section_extraction("TAIC_r_2014_102", "6.3.2")

        assert (
            section
            == """6.3.2.  On 30 January 2015  KiwiRail advised that it had taken the following safety actions:  
 KiwiRail is presently working with a Driver Subject Matter Expert Group in an R&D project 
to help develop a Risk Triggered Commentary Driving procedure that is intended to include 
a stabilis ed approach procedure for non -ETCS trains.  This work is based on international 
work in aviation and other rail domains (RSSB - UK), and is focused on providing an 
enhanced framework for improving LE [train driver] situational awareness and decision -
making .  Once this work has been trialled, it is intended to engage and consult with the 
wider rail industry . 
 

<< Page 14 >>
Page 14 | Final report RO -2014 -102 7. Recommendations"""
        )

    def test_one_early_reference_with_missing_previous_section(self):
        section = self.__test_section_extraction("TAIC_m_2020_202", "4.9")

        assert section is None

    def test_one_reference_in_content_section_with_missing_previous_section(self):
        section = self.__test_section_extraction("TAIC_m_2016_204", "7.5")

        assert section is None

    def test_previous_section_paragraph(self):
        section = ReportSectionExtractor(
            "test report", "test report id"
        )._get_previous_section("5.3.2")
        assert section == "5.3.1"

    def test_previous_section_subsection(self):
        section = ReportSectionExtractor(
            "test report", "test report id"
        )._get_previous_section("5.3")
        assert section == "5.2"

    def test_previous_section_fallback_paragraph(self):
        section = ReportSectionExtractor(
            "test report", "test report id"
        )._get_previous_section("5.3.1")
        assert section == "5.2"

    def test_previous_section_fallback_subsection(self):
        section = ReportSectionExtractor(
            "test report", "test report id"
        )._get_previous_section("5.1")
        assert section == "4"

    def test_prevoius_section_first_section(self):
        section = ReportSectionExtractor(
            "test report", "test report id"
        )._get_previous_section("1.1")

        assert section == "1"


class TestRecommendationExtraction:
    @classmethod
    def setup_class(cls):
        cls.test_data = pd.read_pickle("tests/data/recommendation_test_data.pkl")

    @pytest.mark.parametrize(
        "report_id, expected",
        [
            pytest.param(
                "ATSB_a_2003_980",
                (26, 28),
                id="ATSB_a_2003_980",
            ),
            pytest.param(
                "ATSB_m_2006_234",
                (19, 21),
                id="ATSB_m_2006_234",
            ),
            pytest.param(
                "ATSB_r_2004_004",
                (23, 25),
                id="ATSB_r_2004_004",
            ),
            pytest.param(
                "ATSB_a_2017_105",
                None,
                id="ATSB_a_2017_105",
            ),
            pytest.param(
                "ATSB_r_2014_024",
                (12, 14),
                id="ATSB_r_2014_024",
            ),
            pytest.param(
                "ATSB_a_2002_710",
                (36, 36),
                id="ATSB_a_2002_710 (Safety section is last section)",
            ),
            pytest.param(
                "ATSB_m_2001_163",
                (21, 25),
                id="ATSB_m_2001_163 (No safety action section)",
            ),
        ],
    )
    def test_content_section_reading(self, report_id, expected):
        report_data = self.test_data.loc[report_id]

        extractor = ReportSectionExtractor(
            report_data["text"], report_id, report_data["headers"]
        )
        content_section, _ = extractor.extract_table_of_contents()
        extractor = RecommendationsExtractor(
            report_data["text"], report_id, content_section
        )

        pages = extractor.extract_pages_to_read(content_section).pages

        if expected is None:
            assert len(pages) == 0
        else:
            assert pages[0].start_page == expected[0]
            assert pages[0].end_page == expected[1]

            assert len(pages) == 1

    @pytest.mark.parametrize(
        "report_id",
        [
            pytest.param(
                "ATSB_a_2002_780",
                id="ATSB_a_2002_780 (List of recommendations from other agency)",
            ),
            pytest.param(
                "ATSB_m_2005_215",
                id="ATSB_m_2005_215 (Simple stated recommendations still have recommendation_id)",
            ),
            # Removed temporarily as their is a greater problem with the recommendations missing. See https://dev.azure.com/NZ-TAIC/AI%20at%20TAIC/_sprints/taskboard/AI%20at%20TAIC%20Team/AI%20at%20TAIC/Summer%20AI%20@%20TAIC?workitem=86
            # pytest.param(
            #     "ATSB_a_2021_005",
            #     id="ATSB_a_2021_005 (Modern complete stated recommendations)",
            # ),
            pytest.param(
                "ATSB_m_2008_012",
                id="ATSB_m_2008_012 (No recommendations only safety issues)",
            ),
            pytest.param(
                "ATSB_r_2015_007",
                id="ATSB_r_2015_007 (Modern complete recommendations)",
            ),
        ],
    )
    def test_recommendation_extraction(self, report_id):
        report_data = self.test_data.loc[report_id]

        extractor = RecommendationsExtractor(
            report_data["text"], report_id, report_data["toc"]
        )

        extracted_recommendations = extractor._extract_recommendations_from_text(
            report_data["recommendation_section"]
        )

        expected_recommendations = report_data["recommendations"]

        if expected_recommendations is None:
            assert extracted_recommendations is None
            return
        elif extracted_recommendations is None:
            pytest.fail(f"Expected recommendations for report {report_id} but got None")

        assert len(extracted_recommendations) == len(expected_recommendations)

        print(extracted_recommendations)

        for extracted, expected in zip(
            extracted_recommendations, expected_recommendations
        ):
            assert extracted["recommendation"] == expected["recommendation"].strip()
            assert extracted["recommendation_id"] == expected["recommendation_id"]
            assert (
                SequenceMatcher(
                    None, extracted["recipient"], expected["recipient"]
                ).ratio()
                > 0.8
            )

    @pytest.mark.parametrize(
        "report_id, expected",
        [
            pytest.param(
                "ATSB_a_2014_096",
                0,
                id="ATSB_a_2014_096 (No recommendations)",
            ),
            pytest.param(
                "ATSB_m_2013_011",
                2,
                id="ATSB_m_2013_011 (Modern recommednation)",
            ),
        ],
    )
    def test_complete_process(self, report_id, expected):
        report_data = self.test_data.loc[report_id]

        extractor = RecommendationsExtractor(
            report_data["text"], report_id, report_data["toc"]
        )

        extracted_recommendations, _, _ = extractor.extract_recommendations()

        if expected == 0:
            assert len(extracted_recommendations) == 0
        else:
            assert len(extracted_recommendations) == expected
