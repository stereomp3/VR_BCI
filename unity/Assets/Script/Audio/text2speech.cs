using Meta.WitAi.TTS.Utilities;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

public class text2speech : MonoBehaviour
{
    #region Singleton 
    public static text2speech instance;
    void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of text2speech found!");
            // Destroy(gameObject);
            return;
        }
        else
        {
            // DontDestroyOnLoad(gameObject);
            instance = this;
        }
    }

    #endregion

    public TTSSpeaker _speaker;
    // LSLManager lsl;
    [TextArea]
    public string test_txt;
    public bool is_text = false;
    // Start is called before the first frame update
    void Start()
    {
        // lsl = LSLManager.instance;
        // Speak("The European Union has released plans to recycle all plastic by the year 2030. It wants to ban all types of plastic that can only be used once. The measure comes as a  consequence of China's decision to ban imports of foreign plastic that is to be recycled in in the country. Currently, the EU exports half of its collected plastic,  most of which goes to China.\r\n\r\nThe European Commission also plans to reduce plastic waste that is washed up on North Sea, Atlantic and Mediterranean shores. According to the new proposal, it will be illegal to dump plastic waste in the open seas.\r\n\r\nAlthough the EU does not want to introduce a tax on plastic yet, it does aim at the development and production of new kinds of plastic that can be recycled in Europe. EU countries produce 25  million tons of plastic every year but only a fourth is recycled. It takes plastic hundreds of years to degrade.\r\n\r\nThe EU wants to invest 300 million euros to develop better plastic materials. The new strategy aims at making plastic recycling more profitable.\r\n\r\n \r\n\r\n \r\n\r\nWhile the production of one-time-only usable plastic items, like drinking straws, coffee cups and takeaway packaging is to be reduced, families should also be persuaded to cut down on plastic usage altogether.\r\n\r\nNon-EU countries are also considering cracking down on plastic. Some countries have already started to tax the use of plastic bags. Iceland has announced that it will ban all plastic packaging for domestic products.");
    }

    // Update is called once per frame
    void Update()
    {
        if (is_text)
        {
            Speak(test_txt);
            is_text = false;
        }
    }

    public void Speak(string txt_queue) // on start 觸發 UI 事件， stage01_UIManagers update_coach_text_ui
    {
        if (txt_queue.Length > 200) // 之前使用 249 好像會有問題，試試看縮小範圍
        {
            string[] txts = txt_queue.Split('.');
            for (int i = 0; i < txts.Length; i++)
            {
                _speaker.SpeakQueued(txts[i]);
            }
        }
        else
        {
            _speaker.SpeakQueued(txt_queue);
        }
    }
}
