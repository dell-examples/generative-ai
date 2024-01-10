# Created by scalers.ai for Dell Inc
import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Text

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionLLM(Action):
    def name(self) -> Text:
        return "action_llm"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        url = f"http://api-server:8000/qa/predict?query=\"{urllib.parse.quote(tracker.latest_message['text'])}\""
        try:
            answer = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            return
        tracker.get_slot("context_slot")
        read_answer = answer.read()
        answer_str = json.loads(read_answer.decode("utf8"))
        dispatcher.utter_message(text=answer_str)
        return []
