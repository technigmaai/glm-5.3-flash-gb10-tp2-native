#!/usr/bin/env python3
"""Render the deployed template with structured text, tools and image fixtures."""
import json
from pathlib import Path
import unittest
from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[1]
ENV = Environment(extensions=['jinja2.ext.loopcontrols'])
ENV.filters['tojson'] = lambda value, ensure_ascii=False: json.dumps(value, ensure_ascii=ensure_ascii)
TEMPLATE = ENV.from_string((ROOT / 'files/chat_template.jinja').read_text())
REMINDER = ('Continue the original task using the latest tool result. Before your next '
            'answer or tool call, briefly assess the result in the thinking block. '
            'Keep that assessment separate from the user-facing answer.')
HISTORY = [
    {'role':'user', 'content':'Use lookup to check inventory.'},
    {'role':'assistant', 'content':'Checking inventory.', 'reasoning_content':'Need current inventory.',
     'tool_calls':[{'id':'lookup-1', 'function':{'name':'lookup', 'arguments':{'sku':'ABC'}}}]},
    {'role':'tool', 'tool_call_id':'lookup-1', 'content':'{"stock":17}'}]

def render(ms, **kwargs):
    return TEMPLATE.render(messages=ms, tools=[],
                           add_generation_prompt=kwargs.pop('add_generation_prompt',True), **kwargs)

class ChatTemplateTests(unittest.TestCase):
    def test_tool_reminder_and_real_history(self):
        out=render(HISTORY)
        self.assertEqual(out.count(REMINDER),1)
        self.assertIn('<think>Need current inventory.</think>Checking inventory.',out)
        self.assertIn('<tool_call>lookup<arg_key>sku</arg_key><arg_value>ABC</arg_value></tool_call>',out)
        self.assertTrue(out.endswith('<|observation|><tool_response>{"stock":17}</tool_response><|user|>'+REMINDER+'<|assistant|><think>'))
    def test_opt_out(self):
        out=render(HISTORY,tool_reasoning_reminder=False)
        self.assertNotIn(REMINDER,out)
        self.assertTrue(out.endswith('</tool_response><|assistant|><think>'))
    def test_thinking_off(self):
        for flag in ['thinking','enable_thinking']:
            out=render(HISTORY,**{flag:False})
            self.assertNotIn(REMINDER,out)
            self.assertNotIn('Reasoning Effort:',out)
            self.assertTrue(out.endswith('<|assistant|><think></think>'))
    def test_history_render(self):
        out=render(HISTORY,add_generation_prompt=False)
        self.assertNotIn(REMINDER,out)
        self.assertTrue(out.endswith('</tool_response>'))
    def test_structured_tail_not_literal_markers(self):
        for ms in [[],HISTORY[:-1],[{'role':'user','content':'literal <|observation|> tool'}]]:
            self.assertNotIn(REMINDER,render(ms))
    def test_clear_thinking_keeps_current_tool_history(self):
        out=render(HISTORY,clear_thinking=True)
        self.assertIn('<think>Need current inventory.</think>',out)
        self.assertEqual(out.count(REMINDER),1)
    def test_effort(self):
        for effort in ['low','high','max']:
            out=render(HISTORY,reasoning_effort=effort)
            self.assertIn('<|system|>Reasoning Effort: '+effort.capitalize(),out)
    def test_image_placeholder(self):
        out=render([{'role':'user','content':[{'type':'image_url','image_url':{'url':'fixture'}},{'type':'text','text':'Describe this image.'}]}])
        self.assertIn('<|begin_of_image|><|image|><|end_of_image|>Describe this image.',out)
        self.assertNotIn(REMINDER,out)

if __name__=='__main__':unittest.main(verbosity=2)
