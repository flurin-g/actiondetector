import re

action_item_tags = {"<send_email>",
                    "<create_calendar_entry>",
                    "<create_single_reminder>"}
tag_pattern = r'<\s*[^>]*>|<\s*/\s*a>'
alphanum_pattern = r'[^a-zA-Z0-9\s]'


def remove_tags(text: str) -> str:
    return re.sub(tag_pattern, "", text)


def remove_non_alphanum(text: str) -> str:
    return re.sub(' +', ' ', re.sub(alphanum_pattern, "", text))


def is_action_item(text: str) -> str:
    tags = re.findall(tag_pattern, text)
    return "yes" if any(tag in tags for tag in action_item_tags) else "no"


def parse_line_aimu(line: str) -> dict:
    formatted_line = re.search(r'(.*)\t(.*)\t(.*)', line)
    text = formatted_line.group(3)
    return {
        "speaker": formatted_line.group(2),
        "text": remove_non_alphanum(remove_tags(text)),
        "isActionItem": is_action_item(text)
    }
