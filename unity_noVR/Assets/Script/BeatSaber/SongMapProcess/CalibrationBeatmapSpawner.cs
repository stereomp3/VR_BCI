using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;
using System.Drawing;
using static GameManager;
using UnityEngine.UI;

public class CalibrationBeatmapSpawner : BeatmapSpawner
{
    [Header("Tutorial UI")]
    public Button FinishButtion;
    public Button SkipButtion;
    public GameObject Tutorial_UI;
    // Start is called before the first frame update
    private float stop_time = 0;
    private bool in_tutorial = true;

    private int point = 0; // 測試 MI 砍到幾個 SaberSlicer.cs 會呼叫 onSaberCutCallback，來更新這個
    private int note_num = 10;  // MI trial 後面 random 數量，總 trial 數量為 ME 4 (數量根據使用者成功率有所變動) + MI 4 + MI 2 (都是左右) + note_num (random) 
    void Start()
    {
        GM = GameManager.instance;
        GM.onSaberCutCallback += add_the_MI_point;
        GM.onGameStopCallback += stop_gmae;
        GM.onGameStartCallback += start_gmae;

        StartCoroutine(RunTutorial());
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    public enum TutorialStep
    {
        Step1_ClickButton,
        Step2_MoveCharacter,
        Step3_CollectItem,
        Step4_Finish,
    }
    IEnumerator RunTutorial()
    {
        // ReceiveLSLMarker.instance.is_simulated = true;  // set lsl to simulate
        
        TCP_Client.instance.is_simulated = true;  // set lsl to simulate
        yield return new WaitForSeconds(2.5f); // wait for connection
        GM.use_LSL_to_controll_saber = false;  // don't use lsl to controll
        if (Config.pass_tutorial) // MI 總數 20，前面 6 run 為 val // 20//3 = 6.66
        {
            GM.set_autosaber();  // switch to MI
            // yield return spawn_with_old_beatmap(set_random_map(6)); // for quick test
            yield return spawn_with_old_beatmap(set_random_map(20));
            // yield return spawn_with_old_beatmap(set_random_map(8));
            // yield return spawn_with_old_beatmap(set_random_map(12));
        }
        else
        {
            start_tutorial();
            text2speech.instance.Speak("The system will now enter tutorial mode.");
            bool clicked = false;
            FinishButtion.onClick.AddListener(() => clicked = end_tutorial());  // return true
            SkipButtion.onClick.AddListener(() => clicked = end_tutorial());
            yield return new WaitUntil(() => clicked);
            yield return new WaitForSeconds(1f);
            GM.set_autosaber();  // switch to MI
            text2speech.instance.Speak("Now, let's start Motor Imagery");
            yield return new WaitForSeconds(3f);
            yield return spawn_with_old_beatmap(set_random_map(20));
            /*text2speech.instance.Speak("A long note will now appear on the right side and left side. Use your hand to slash it.");
            yield return new WaitForSeconds(5f);
            yield return StartCoroutine(spawn_two_trial(6));  // 目前一個 note 會有 5 個 point，左右各會有一個，預計 ME 要砍到 6 個才能到下一個
            yield return StartCoroutine(spawn_two_trial(8));  // ME trial
            text2speech.instance.Speak("Now, let's start Motor Imagery");
            yield return new WaitForSeconds(3f);
            GM.set_autosaber();  // switch to MI
            yield return StartCoroutine(spawn_two_trial(0));  // MI trial
            yield return StartCoroutine(spawn_two_trial(0));  // MI trial

            yield return StartCoroutine(spawn_two_trial_fail_one(0)); // MI trial fail 1

            yield return spawn_with_old_beatmap(set_random_map(note_num)); // spawn random note_num note    */
        }

        // training
        text2speech.instance.Speak("Done! Now we’ll send the data for calibration.");
        GM.start_training();  

        //text2speech.instance.Speak("A long note will now appear on the left side, with a cube on top. Please use your left hand to slash it.");

        //text2speech.instance.Speak("A long note will now appear on the right side, with a cube on top. Use your hand to slash it.");

        //text2speech.instance.Speak("Remember the feeling of using your hand to slash the cube. Now, try slashing it again.");

        //text2speech.instance.Speak("Now, let's try using motor imagery to slash the cube.");

        //text2speech.instance.Speak("Try silently saying 'left' or 'right' in your mind as you slash the cube — it can help with the imagery.");

        //text2speech.instance.Speak("Another useful tip is to imagine moving your left or right hand without actually moving it — this can strengthen your motor imagery.");

        //text2speech.instance.Speak("If it's really hard to imagine, alternatively, you can slightly tense the muscles in your left or right hand to enhance the vividness of your motor imagery.");



        // 結束
        // tutorialText.text = "教學完成！";
        in_tutorial = false;

        /*TtsClient.instance.Speak("開始教學模式");
        TtsClient.instance.Speak("這個遊戲主要是要使用運動想像來操控光劍砍方塊");
        TtsClient.instance.Speak("現在左邊會出現長條音符，上面會有方塊，請使用手揮砍他");
        TtsClient.instance.Speak("現在出現在右邊，請使用手揮砍他");
        TtsClient.instance.Speak("記住砍方塊動手的感覺，現在請你再次嘗試砍方塊");
        TtsClient.instance.Speak("現在，讓我們嘗試使用運動想像砍方塊");
        TtsClient.instance.Speak("嘗試在砍方塊的時候心裡默念左或是右，可以幫助想像");
        TtsClient.instance.Speak("還有一個小技巧是，可以透過想像左手或是右手運動，但是不動，來增強想像");
        TtsClient.instance.Speak("如果真的很難想像，可以左手或是右手肌肉輕微用力，來達到強烈想像的效果");*/

        // Step 1: 點擊按鈕
        // tutorialText.text = "請點擊開始按鈕";
        // bool clicked = false;
        // actionButton.onClick.AddListener(() => clicked = true);
        // yield return new WaitUntil(() => clicked);

        // currentStep = TutorialStep.Step2_MoveCharacter;

        // Step 2: 移動角色
        // tutorialText.text = "請使用 WASD 移動角色";
        // yield return new WaitUntil(() => PlayerHasMoved()); 要寫 funtion 必須要 return true

        // currentStep = TutorialStep.Step3_CollectItem;

        // Step 3: 收集物品
        // tutorialText.text = "請撿起前方的物品";
        // yield return new WaitUntil(() => itemCollected);

        // currentStep = TutorialStep.Step4_Finish;

        
    }
    protected OldBeatmap create_olde_beatmap(int data_count)
    {
        OldBeatmap data = new OldBeatmap();
        data.notes = new List<OldNote>();

        // 根據 data_count 來創建指定數量的 OldNote
        for (int i = 0; i < data_count; i++)
        {
            OldNote note = new OldNote();
            data.notes.Add(note);
        }
        return data;
    }
    IEnumerator spawn_two_trial(int expect_point) // expect_point 為了 ME 設計，但這個 trial 也可以用於 MI
    {
        yield return StartCoroutine(spawn_right_note());
        yield return new WaitForSeconds(7f);
        yield return StartCoroutine(spawn_left_note());
        yield return new WaitForSeconds(5f);
        while (point < expect_point) // ME 分數沒到，重新
        {
            text2speech.instance.Speak("please try again.");
            yield return new WaitForSeconds(2f);
            point = 0;
            yield return StartCoroutine(spawn_right_note());
            yield return new WaitForSeconds(7f);
            yield return StartCoroutine(spawn_left_note());
            yield return new WaitForSeconds(5f);
        }
        point = 0;
    }
    IEnumerator spawn_two_trial_fail_one(int fail) // 1 is always left (fail right), 0 is right (fail left)
    {
        GM.use_LSL_to_controll_saber = true;  // fail one trial on two trial
        // ReceiveLSLMarker.instance.simulatedInput = fail;  // 預設是 0
        TCP_Client.instance.simulatedInput = fail;  // 改成使用 TCP 
        yield return StartCoroutine(spawn_two_trial(0)); // MI trial

        GM.use_LSL_to_controll_saber = false; // back to correct hit
        // 根據預設 0 說出對應內容
        if(fail == 0)
        {
            text2speech.instance.Speak("When imagining the movement of the saber with your left hand, if you mistakenly picture it as your right hand, the saber will swing incorrectly.");
        }
        yield return new WaitForSeconds(7f);
    }

    IEnumerator spawn_with_old_beatmap(OldBeatmap data)
    {
        foreach (OldNote note in data.notes)
        {
            SpawnOldBlock(note);
            if (GM.is_audosaber)
            {
                int x = note.lineIndex, y = note.lineLayer, d = note.cutDirection;
                if (GM.onAutoSaberCallback != null) StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, false));
            }
            yield return new WaitForSeconds(7f);
        }
    }
    void add_the_MI_point()
    {
        point += 1;
    }
    IEnumerator spawn_right_note()
    {
        yield return StartCoroutine(check_is_pause());
        SpawnOldBlock(right_blue_note());
        if (GM.is_audosaber)
        {
            int x = 2, y = 0, d = 0;
            if (GM.onAutoSaberCallback != null) StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, false));
        }
        yield return null;
    }

    IEnumerator spawn_left_note()
    {
        yield return StartCoroutine(check_is_pause());
        SpawnOldBlock(left_red_note());
        if (GM.is_audosaber)
        {
            int x = 1, y = 0, d = 0;
            if (GM.onAutoSaberCallback != null) StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, false));
        }
        yield return null;
    }

    IEnumerator check_is_pause() // 在生成之前先確認有沒有暫停
    {
        while (is_pause) {
            yield return null;
        }
    }
    OldNote left_red_note()
    {
        OldNote note = new OldNote();

        note.lineIndex = 1;  //  from 0 (left) to 3 (right)
        note.lineLayer = 0; //  from 0 (bottom) to 2 (top)

        note.time = 0; // Beat time
        note.type = 0; // color (0 = left (red), 1 = right (blue))
        note.cutDirection = 0; // direction
        return note;
    }
    OldNote right_blue_note()
    {
        OldNote note = new OldNote();

        note.lineIndex = 2;  //  from 0 (left) to 3 (right)
        note.lineLayer = 0; //  from 0 (bottom) to 2 (top)

        note.time = 0; // Beat time
        note.type = 1; // color (0 = left (red), 1 = right (blue))
        note.cutDirection = 0; // direction
        return note;
    }

    OldBeatmap set_random_map(int note_num)
    {
        OldBeatmap data = create_olde_beatmap(note_num);
        return set_random_old_note(data);
    }

    bool end_tutorial() // () => clicked = true，() 為輸入，最後為 return true 然後給到 clicked
    {
        Tutorial_UI.SetActive(false);
        return true;    
    }

    void start_tutorial() 
    {
        Tutorial_UI.SetActive(true);
    }
}
