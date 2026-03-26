import pyttsx3  # pip install pyttsx3

def init_tts():  # 初始化语音合成引擎
    """
    初始化语音合成引擎
    :return:
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # 设置语速
        engine.setProperty('volume', 1.0)  # 设置音量
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'zh' in voice.id.lower() or 'chinese' in voice.id.lower():
                engine.setProperty('voice', voice.id)
                break
        return engine
    except Exception as e:
        print(f"初始化语音合成引擎失败: {e}")
        return None

def test_tts():  # 测试语音合成
    engine = init_tts()
    if not engine:
        print("无法初始化语音合成引擎")
        return
    text = "您已进入工地施工场所，请规范佩戴安全帽，穿好反光衣！！！"
    try:
        engine.say(text)
        engine.runAndWait()
        print("语音合成完成")
    except Exception as e:
        print(f"语音合成失败: {e}")

if __name__ == "__main__":
    test_tts()