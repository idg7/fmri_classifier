import pandas as pd
import re


def script2speech(script: str) -> pd.DataFrame:
    script = script.replace('\t', '').replace('\n'+15*' ','\n')
    
    dataset = {
        'character': [],
        'speech': []
    }
    
    for match in re.finditer(r'(\n +[A-Z (.)]+\n)', script):
        cls = match.group(0)
        cls = cls.replace('\n', '').replace(22*' ', '')
        
        speech_start_idx = match.end(0)
        speech_end_idx = script.find('\n\n', speech_start_idx)
        speech = script[speech_start_idx : speech_end_idx]
        speech = speech.replace('\n', '').replace(10*' ', '')
        
        dataset['character'].append(cls)
        dataset['speech'].append(speech)
        
    return pd.DataFrame(dataset)
    
