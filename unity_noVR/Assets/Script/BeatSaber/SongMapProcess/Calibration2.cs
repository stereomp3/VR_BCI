using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using TMPro;
using UnityEngine;

public class Calibration2 : BeatmapSpawner
{
    [Header("Task UI")]
    public TextMeshProUGUI checkpoint_text;

    float base_offset;
    AutoSaber AS;
    int pre_wrong_hit;
    float wait_time = 2f;
    TimeLogger TL;
    public int TTL = 34; // 當這個數值變成 0，就會強制結束，把時間控制在 15 分鐘內，詳細過程可以去看 beat saber note README。
    // Start is called before the first frame update
    void Start()
    {
        GM = GameManager.instance;
        GM.onGameStopCallback += stop_gmae;
        GM.onGameStartCallback += start_gmae;

        TC = TCP_Client.instance;

        beatSaberInfoLoader = GetComponent<BeatSaberInfoLoader>(); // select difficulty
        song = beatSaberInfoLoader.song;
        bpm = beatSaberInfoLoader.song_bpm;
        beatsPerSecond = bpm / 60f;

        per_spawn_second = GM.song_delay_time;
        start_game_seconds = GM.start_game_seconds;

        load_song();

        songStartTime = Time.time;
        base_offset = 1f / ((120 / bpm) * (120 / bpm)) * 60 / bpm;
        Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ base_offset: " + base_offset);

        StartCoroutine(RunTutorial());

        AS = AutoSaber.instance;
        pre_wrong_hit = AS.wrong_hit;

        TL = TimeLogger.instance;   
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    IEnumerator RunTutorial()
    {
        // ReceiveLSLMarker.instance.is_simulated = true;  // set lsl to simulate

        // TCP_Client.instance.is_simulated = true;  // set lsl to simulate
        yield return new WaitForSeconds(3.5f); // wait for connection
        // GM.use_LSL_to_controll_saber = false;  // don't use lsl to controll

        checkpoint_text.text = "0 / 5 \nTTL :  " + TTL.ToString();
        // 先收集 4 個 trial 給到 python buffer
        yield return StartCoroutine(spawn_Nb_random_trial(2));

        checkpoint_text.text = "1 / 5 \nTTL :  " + TTL.ToString();
        text2speech.instance.Speak("Now Let's try to imagine moving your right hand.");

        yield return StartCoroutine(send_info_and_wait());

       
        // 右右
        yield return StartCoroutine(spawn_N_trial(new int[] { 0, 0 }));
        if (AS.wrong_hit == pre_wrong_hit && TTL > 0)
        {
            checkpoint_text.text = "2 / 5 \nTTL :  " + TTL.ToString();
            text2speech.instance.Speak("Try to imagine moving your left hand.");
        }
        else checkpoint_text.text = "1 / 5 \nTTL :  " + TTL.ToString();
        yield return StartCoroutine(send_info_and_wait());

        while (AS.wrong_hit > pre_wrong_hit)  // 有錯誤就重新
        {
            yield return StartCoroutine(spawn_N_trial(new int[] { 0, 0 }));
            if (AS.wrong_hit == pre_wrong_hit && TTL > 0)
            {
                checkpoint_text.text = "2 / 5 \nTTL :  " + TTL.ToString();
                text2speech.instance.Speak("Try to imagine moving your left hand.");
            }
            else checkpoint_text.text = "1 / 5 \nTTL :  " + TTL.ToString();
            yield return StartCoroutine(send_info_and_wait());
        }

        // 左左
        yield return StartCoroutine(spawn_N_trial(new int[] { 1, 1 }));
        if (AS.wrong_hit == pre_wrong_hit && TTL > 0)
        {
            checkpoint_text.text = "3 / 5 \nTTL :  " + TTL.ToString();
            text2speech.instance.Speak("Greate! Let's try to imagine moving your left and right hand.");
        }
        else checkpoint_text.text = "2 / 5 \nTTL :  " + TTL.ToString();
        yield return StartCoroutine(send_info_and_wait());

        while (AS.wrong_hit > pre_wrong_hit)  // 有錯誤就重新
        {
            yield return StartCoroutine(spawn_N_trial(new int[] { 1, 1 }));
            if (AS.wrong_hit == pre_wrong_hit && TTL > 0)
            {
                checkpoint_text.text = "3 / 5 \nTTL :  " + TTL.ToString();
                text2speech.instance.Speak("Greate! Let's try to imagine moving your left and right hand.");
            }
            else checkpoint_text.text = "2 / 5 \nTTL :  " + TTL.ToString();
            yield return StartCoroutine(send_info_and_wait());
        }

        // 右左
        yield return StartCoroutine(spawn_N_trial(new int[] { 0, 1 }));
        if (AS.wrong_hit == pre_wrong_hit && TTL > 0)
        {
            checkpoint_text.text = "4 / 5 \nTTL :  " + TTL.ToString();
            text2speech.instance.Speak("The next task will be more challenging. Please stay focused.");
        }
        else checkpoint_text.text = "3 / 5 \nTTL :  " + TTL.ToString();
        yield return StartCoroutine(send_info_and_wait());

        while (AS.wrong_hit > pre_wrong_hit)  // 有錯誤就重新
        {
            yield return StartCoroutine(spawn_N_trial(new int[] { 0, 1 }));
            if(AS.wrong_hit == pre_wrong_hit && TTL > 0)
            {
                checkpoint_text.text = "4 / 5 \nTTL :  " + TTL.ToString();
                text2speech.instance.Speak("The next task will be more challenging. Please stay focused.");
            }
            else checkpoint_text.text = "3 / 5 \nTTL :  " + TTL.ToString();
            yield return StartCoroutine(send_info_and_wait());
        }

        

        // 左右 * 4 random
        yield return StartCoroutine(spawn_Nb_random_trial(4));

        if (AS.wrong_hit != pre_wrong_hit) checkpoint_text.text = "4 / 5 \nTTL :  " + TTL.ToString();

        yield return StartCoroutine(send_info_and_wait());
        while (AS.wrong_hit > pre_wrong_hit)  // 有錯誤就重新
        {
            yield return StartCoroutine(spawn_Nb_random_trial(4));
            if (AS.wrong_hit != pre_wrong_hit) checkpoint_text.text = "4 / 5 \nTTL :  " + TTL.ToString();
            if (AS.wrong_hit == pre_wrong_hit) break;// 最後如果成功就直接退出

            yield return StartCoroutine(send_info_and_wait());
        }

        
        checkpoint_text.text = "5 / 5";
        text2speech.instance.Speak("Well done! You did a great job.");
        StartCoroutine(GM.SetFinalAcc());

        /*if (Config.pass_tutorial) // MI 總數 20，前面 6 run 為 val // 20//3 = 6.66
        {
            // GM.set_autosaber();  // switch to MI
            // yield return spawn_with_old_beatmap(set_random_map(6)); // for quick test
            // yield return spawn_with_old_beatmap(set_random_map(20));
            yield return spawn_right_note();
            // yield return spawn_with_old_beatmap(set_random_map(8));
            // yield return spawn_with_old_beatmap(set_random_map(12));
        }*/
        // training
        // text2speech.instance.Speak("Done! Now we’ll send the data for calibration.");
        // GM.start_training();

        //text2speech.instance.Speak("A long note will now appear on the left side, with a cube on top. Please use your left hand to slash it.");

        //text2speech.instance.Speak("A long note will now appear on the right side, with a cube on top. Use your hand to slash it.");

        //text2speech.instance.Speak("Remember the feeling of using your hand to slash the cube. Now, try slashing it again.");

        //text2speech.instance.Speak("Now, let's try using motor imagery to slash the cube.");

        //text2speech.instance.Speak("Try silently saying 'left' or 'right' in your mind as you slash the cube — it can help with the imagery.");

        //text2speech.instance.Speak("Another useful tip is to imagine moving your left or right hand without actually moving it — this can strengthen your motor imagery.");

        //text2speech.instance.Speak("If it's really hard to imagine, alternatively, you can slightly tense the muscles in your left or right hand to enhance the vividness of your motor imagery.");

    }

    IEnumerator waiting_loop()
    {
        if(TC.is_simulated) yield return new WaitForSeconds(0.5f);
        else
        {
            yield return new WaitForSeconds(0.5f);
            if (!GM.calibration_model_finish) yield return StartCoroutine(waiting_loop());
            else GM.calibration_model_finish = false;
        }
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
    IEnumerator spawn_N_trial(int[] trials) // 產生 N 個 trial，填入 0 代表 right， 1 代表 left。
    {
        pre_wrong_hit = AS.wrong_hit;

        if (TTL <= 0) yield break; // 強制退出

        foreach (int trial in trials)
        {
            if (trial == 0) yield return StartCoroutine(spawn_right_note());
            else yield return StartCoroutine(spawn_left_note());
            yield return new WaitForSeconds(wait_time);
        }
        
        TTL -= (int)(trials.Length / 2);
        yield return new WaitForSeconds(wait_time);
    }

    IEnumerator spawn_Nb_random_trial(int n) // 產生 N 個 trial random balance 的 trial，輸入 int n。
    {
        pre_wrong_hit = AS.wrong_hit;

        if (TTL <= 0) yield break; // 強制退出

        yield return StartCoroutine(spawn_with_old_beatmap(set_random_map(n)));

        TTL -= (int)(n / 2);
        yield return new WaitForSeconds(wait_time);
    }

    IEnumerator send_info_and_wait()
    {
        TC.send_string_to_python(Config.send_python_tcp_calibration_start);
        // TL.reset_trialCount();
        yield return StartCoroutine(waiting_loop());
    }

    IEnumerator spawn_with_old_beatmap(OldBeatmap data)
    {
        foreach (OldNote note in data.notes)
        {
            int x = note.lineIndex;
            if (x == 1) yield return StartCoroutine(spawn_left_note());
            else yield return StartCoroutine(spawn_right_note());

            yield return new WaitForSeconds(wait_time);
        }
    }
    IEnumerator spawn_right_note()
    {
        yield return StartCoroutine(spawn_right_note_one(is_first: true));
        yield return StartCoroutine(spawn_right_note_one());
        yield return StartCoroutine(spawn_right_note_one());
        yield return StartCoroutine(spawn_right_note_one());
        yield return StartCoroutine(spawn_right_note_one(is_end: true));
    }
    IEnumerator spawn_right_note_one(bool is_first = false, bool is_end = false)
    {
        yield return StartCoroutine(check_is_pause());
        if (is_first) SpawnLine(right_blue_note(), beatThreshold_up, assign_dur: base_offset * 12);
        SpawnOldBlock(right_blue_note());
        if (GM.is_audosaber)
        {
            int x = 2, y = 0, d = 0;
            if (GM.onAutoSaberCallback != null) StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, is_end));
        }
        yield return new WaitForSeconds(base_offset);
    }
    IEnumerator spawn_left_note()
    {
        yield return StartCoroutine(spawn_left_note_one(is_first: true));
        yield return StartCoroutine(spawn_left_note_one());
        yield return StartCoroutine(spawn_left_note_one());
        yield return StartCoroutine(spawn_left_note_one());
        yield return StartCoroutine(spawn_left_note_one(is_end: true));
    }
    IEnumerator spawn_left_note_one(bool is_first = false, bool is_end = false)
    {
        yield return StartCoroutine(check_is_pause());
        if (is_first) SpawnLine(left_red_note(), beatThreshold_up, assign_dur: base_offset * 12);
        SpawnOldBlock(left_red_note());
        if (GM.is_audosaber)
        {
            int x = 1, y = 0, d = 0;
            if (GM.onAutoSaberCallback != null) StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, is_end));
        }
        yield return new WaitForSeconds(base_offset);
    }

    IEnumerator check_is_pause() // 在生成之前先確認有沒有暫停
    {
        while (is_pause)
        {
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
}
