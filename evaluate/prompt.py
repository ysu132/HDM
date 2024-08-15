import re
from termcolor import colored


def parse_and_check_numerical_answer(answer, min=None, max=None):
    attempt = parse_numerical_answer(answer, min, max)
    if attempt is not None:
        if attempt < min or attempt > max:
            return None
        return attempt

    return None


def parse_numerical_answer(answer, min=None, max=None):
    # get all numbers in a string
    numbers = re.findall(r'\d+', answer)
    if len(numbers) == 1:
        return int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    return None


def validate_number(x, min=0, max=100):
    attempt = parse_and_check_numerical_answer(x, min, max)
    if attempt is not None:
        return attempt
    return None


def parse_classes(answer, classes):
    final_class = None
    for i in range(len(classes)):
        if classes[i].lower() in answer.lower():
            if final_class is None:
                final_class = i
            else:
                print(colored(f"Two classes found in answer {answer}", "red"))
                return None

    return final_class


def validate_stars(x):
    x = x.lower()
    # try to find all possible answers as sometimes it seems to be explaining itself
    possible_answers = set()

    # check if string x contains * characters
    if "*" in x:
        possible_answers.add(x.count("*"))
    if "★" in x:
        possible_answers.add(x.count("★"))

    x = f" {x} ".replace("\n", " ")
    # possible answers: "five stars", "5 stars", "five", "five starts: perfect translation", ...
    if " one " in x or "1 star" in x:
        possible_answers.add(1)
    if " two " in x or "2 star" in x:
        possible_answers.add(2)
    if " three " in x or "3 star" in x:
        possible_answers.add(3)
    if " four " in x or "4 star" in x:
        possible_answers.add(4)
    if " five " in x or "5 star" in x:
        possible_answers.add(5)

    numerical = parse_numerical_answer(x)
    if numerical is not None:
        possible_answers.add(numerical)

    if len(possible_answers) == 1:
        answer = possible_answers.pop()
        if 1 <= answer <= 5:
            return answer
    return None


language_codes = {
    "en": "English",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}

prompts = {
    "GEMBA-DA": {
        "prompt": 'Score the following sentence on a continuous scale from 0 to 100, where a score of zero means "no humorous" and score of one hundred means "perfect humourous".\n\nSentence: "{sentence}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-DA_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: {reference_seg}\n{target_lang} machine translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True},

    "GEMBA-SQM": {
        "prompt": 'Score the following sentence on a continuous scale from 0 to 100 that starts on "No humour", goes through "Some humour", then "Humour", up to "Most humour".\n\nSentence: "{sentence}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-SQM-fl": {
        "prompt": 'Score the following sentence on a continuous scale from 0 to 100 that starts on "No fluency", goes through "Some fluency", then "Most fluency", up to "Perfect fluency".\n\nSentence: "{sentence}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},
    "GEMBA-SQM-in": {
        "prompt": 'Score the following {lang} sentence on a continuous scale from 0 to 100 that starts on "No coherency", goes through "Some coherency", then "Most coherency", up to "Perfect coherency".\n\nSentence: "{sentence}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False},

    "GEMBA-stars": {
        "prompt": 'Score the following sentence with one to five stars. Where one star means "No humour", two stars mean "less humour", three stars mean "Some humour", four stars mean "Most humour", and five stars mean "Perfect humour".\n\nSentence: "{sentence}"\nStars: (Only output the number of stars)',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": False},
    "GEMBA-stars-fl": {
        "prompt": 'Score the following sentence with one to five stars. Where one star means "No fluency", two stars mean "less fluency", three stars mean "Some fluency", four stars mean "Most fluency", and five stars mean "Perfect fluency".\n\nSentence: "{sentence}"\nStars: (Only output the number of stars)',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": False},
    "GEMBA-stars-in": {
        "prompt": 'Score the following sentence with one to five stars. Where one star means "No coherency", two stars mean "less coherency", three stars mean "Some coherency", four stars mean "Most coherency", and five stars mean "Perfect coherency".\n\nSentence: "{sentence}"\nStars: (Only output the number of stars)',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": False},

    "GEMBA-stars_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to the human reference with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": True},

    "GEMBA-classes": {
        "prompt": 'Classify the quality of following sentence into one of following classes: "No humorous", "less humorous", "Some humorous", "Most humorous", "Perfect humorous".\n\n Sentence: "{sentence}"\nClass:',
        "use_ref": False,
        "validate_answer": lambda x, classes=["No humorous", "less humorous", "Some humorous", "Most humorous", "Perfect humorous"]: parse_classes(x, classes),
        "max_tokens": 100},

    "GEMBA-classes-fl": {
        "prompt": 'Classify the quality of following sentence into one of following classes: "No coherency and fluency", "Less coherency and fluency", "Some coherency and fluency", "Most coherency and fluency", "Perfect coherency and fluency".\n\n Sentence: "{sentence}"\nClass:',
        "use_ref": False,
        "validate_answer": lambda x, classes=["No coherency and fluency", "Less coherency and fluency", "Some coherency and fluency", "Most coherency and fluency", "Perfect coherency and fluency"]: parse_classes(x, classes),
        "max_tokens": 100},
}
