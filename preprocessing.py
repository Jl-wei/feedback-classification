from string import digits
import re


# convert text to lower case
def to_lowercase(x):
    if type(x) is str:
        return x.lower()
    elif x is None:
        return str(None)


# remove urls (e.g. www.sap.com https:)
def remove_urls(x):
    URL_FAST_REGEX = "(((https?|ftp|file):\/\/)|www\\.)\\S+"
    return re.sub(URL_FAST_REGEX, " ", x)


# remove tags (e.g. < br> <html>)
def remove_tags(x):
    FILTER_ALL_TAGS = "<[^>]*>"
    return re.sub(FILTER_ALL_TAGS, " ", x)


# remove special characters (eg. ' ! Ã £ Ã ¢ { })
def filter_characters_regex(x):
    return re.sub(r"[^a-zA-Z0-9,.']+", " ", x)


def numbers_and_mail(txt):
    # Remove Number and Email Id
    pretxt = re.sub(r"\S*@\S*\s?|\w*\d\w*", "", str(txt))
    # pretxt = re.sub(r'[\w\.,]+@[\w\.,]+\.\w+','' ,txt)
    remove_digits = str.maketrans("", "", digits)
    pretxt = pretxt.translate(remove_digits)
    return pretxt


# remove multiple whitespaces
def remove_multiple_white_spaces(x):
    FILTER_MULTIPLE_WHITESPACES = "\s\s+"
    return re.sub(FILTER_MULTIPLE_WHITESPACES, " ", x)


# strip spaces at the end and the starting
def strip_white_spaces(x):
    return x.strip()


# invoke all above functions
def preprocess(x):
    return strip_white_spaces(
        remove_multiple_white_spaces(
            numbers_and_mail(
                filter_characters_regex(remove_tags(remove_urls(to_lowercase(x))))
            )
        )
    )
