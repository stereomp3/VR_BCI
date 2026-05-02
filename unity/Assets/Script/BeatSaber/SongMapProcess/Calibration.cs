using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using static UnityEditor.Experimental.AssetDatabaseExperimental.AssetDatabaseCounters;

public class Calibration : BeatmapSpawner // 用於 Calibration 邏輯，70 % 以上的正向回饋
{
    int base_score = 65;
    AutoSaber AS;
    int pre_correct_hit;
    int pre_wrong_hit;
    bool is_sham = false; // 判斷是否自動衝能
    int current_x = 0;
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

        AS = AutoSaber.instance;
        pre_correct_hit = AS.correct_hit;
        pre_wrong_hit = AS.wrong_hit;
    }

    void Update()
    {
        if (is_sham)
        {
            if (currentNoteIndex < old_notes.Count)
            {
                ModifyTCPBuffer(current_x);
            }
        }
        if (is_pause)
        {
            songDelayTime += Time.deltaTime;
        }
        else
        {
            if (start_game_seconds < 0)
            {
                float elapsedTime = Time.time - songStartTime - songDelayTime;
                float currentBeat = elapsedTime * beatsPerSecond;
                if (currentNoteIndex >= old_notes.Count)
                {
                    if (!beatSaberInfoLoader.is_end_song)
                    {
                        StartCoroutine(GM.SetFinalAcc());
                        StartCoroutine(ECEOCondition(true)); // 睜眼閉眼任務，與是否訓練
                    }
                    beatSaberInfoLoader.is_end_song = true;
                    return;
                }

                while (currentNoteIndex < old_notes.Count && old_notes[currentNoteIndex].time <= currentBeat)
                {
                    SpawnOldBlock(old_notes[currentNoteIndex]);
                    if (GM.is_audosaber)
                    {
                        int x = old_notes[currentNoteIndex].lineIndex, y = old_notes[currentNoteIndex].lineLayer, d = old_notes[currentNoteIndex].cutDirection;
                        if (GM.onAutoSaberCallback != null)
                        {
                            if (finalNote.Contains(old_notes[currentNoteIndex]) || Config.group_note_num <= 1) // group 小於等於 1，就直接全部腦波控制
                            {
                                StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, true));
                                is_sent_line = true;
                                StartCoroutine(WaitForCaculateScore());
                            }
                            else
                            {
                                StartCoroutine(GM.onAutoSaberCallback.Invoke(x, y, d, false));
                                if (is_sent_line)
                                {
                                    SpawnLine(old_notes[currentNoteIndex], beatThreshold_up);
                                    is_sent_line = false;
                                }
                            }
                        }
                        if (GM.setupAutoSaberCallback != null) GM.setupAutoSaberCallback.Invoke();
                    }
                    currentNoteIndex++;
                }
            }
            else
            {
                start_game_seconds -= Time.deltaTime;
                songStartTime = Time.time;
            }
        }
    }
    IEnumerator WaitForCaculateScore()
    {
        yield return new WaitForSeconds(per_spawn_second);
        if (AS.correct_hit > pre_correct_hit) // correct
        {
            base_score += 1;
        }
        else if (AS.wrong_hit > pre_wrong_hit) // wrong
        {
            base_score -= 3;
        }
        if (base_score < 65)
        {
            is_sham = true;
        }
        else
        {
            is_sham = false;
        }
        // Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ base_score: " + base_score + ", AS.correct_hit: " + AS.correct_hit + ", AS.wrong_hit: " + AS.wrong_hit + ", is_sham: " + is_sham);
        pre_correct_hit = AS.correct_hit;
        pre_wrong_hit = AS.wrong_hit;
        current_x = old_notes[currentNoteIndex].lineIndex;
    }

    void ModifyTCPBuffer(int x)  // 類似 ShameFeedBack，增加資料到 Buffer 裡面，讓結果會正確，判斷內容與 AutoSaber.cs 一致
    {
        int n = TC.valueHistory.Count;
        int count1 = GetBuffer1(); // left
        int count0 = n - count1;  // right
        // Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ count1:" + count1 + ", count0: " + count0 + ", x: " + x);
        if (x == 1) // left
        {
            if(count1 <= n * 0.7f)
            {
                TC.update_history(1);
            }
        }
        else  // x = 2, right
        {
            if (count0 <= n * 0.7f)
            {
                TC.update_history(0);
            }
        }
    }

    int GetBuffer1() // 會給回 buffer 裡面 1 有的數量
    {
        int count1 = 0;
        foreach (var item in TC.valueHistory)
        {
            if (item == 1)
                count1 += 1;
        }
        return count1;
    }
}
