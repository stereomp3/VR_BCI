using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using static Oculus.Interaction.Context;

public class AutoSaber : MonoBehaviour
{
    public static AutoSaber instance;
    public Transform saberL;
    public Transform saberR;
    public float spawnZ = 5f;
    public float offset_x = 2f;
    public float scale_x = 0.3f;
    public float offset_y = 2f;
    public float swingDuration = 0.3f; // seconds to swing
    public float delay_time = 2.5f;  // forward.cs 移動時間總共 1.5
    public float swingAngleZ = 60f; // 
    public float swingAngleY = 45f; // 主要動這個，揮動角度
    public float swing_time = 1f; // 揮動次數

    public int correct_hit = 0;
    public int wrong_hit = 0;

    private GameManager GM;
    // private ReceiveLSLMarker lSLMarker;
    private TCP_Client tcp_predict;
    private Transform saber;
    private void Awake()
    {
        if (instance != null)
        {
            Debug.LogWarning("More than one instance of AutoSaber found!");
            Destroy(gameObject);
            return;
        }
        else
        {
            // DontDestroyOnLoad(gameObject);
            instance = this;
        }
    }
    private void Start()
    {
        // lSLMarker = ReceiveLSLMarker.instance;
        tcp_predict = TCP_Client.instance;
        GM = GameManager.instance;
        GM.onAutoSaberCallback += MoveAndSwingSaberToNote;
    }
    public IEnumerator MoveAndSwingSaberToNote(int note_x, int note_y, int note_direction, bool use_brain_to_controll) // 目前在 BeatmapSpawner 觸發 (update)
    {
        yield return new WaitForSeconds(delay_time);
        if (note_x == 1) saber = saberL;
        else saber = saberR;
        
        // 1. 計算世界座標位置
        Vector3 targetPos = new Vector3(note_x * scale_x - offset_x, note_y - offset_y, spawnZ);
        // 順便旋轉劍
        Quaternion startRot = saber.rotation;
        Quaternion endRot = startRot * GetSwingYaw(note_direction);
        // 2. 將 Saber 移動到 note 上方
        //  saber.position = targetPos + GetStartOffset(note_direction);
        // saber.rotation = GetSwingRotation(note_direction);

        // 3. 揮砍動畫（移動到 note 中心）
        for (int i = 0; i < swing_time; i++)
        {
            if (use_brain_to_controll)  // 目前設定 group 最後一個為用腦波控制
            {
                if (GM.use_LSL_to_controll_saber)
                {
                    int count1 = 0;
                    float totalWeight = 0f;

                    float minWeight = 1f;   // 最舊資料的權重
                    float maxWeight = 5f;  // 最新資料的權重
                    bool useExponential = false; // true = 指數遞增, false = 線性遞增

                    int ii = 0;
                    int n = tcp_predict.valueHistory.Count;
                    foreach (var item in tcp_predict.valueHistory)
                    {
                        float tt = (float)ii / (n - 1);
                        float weight = useExponential
                            ? Mathf.Lerp(minWeight, maxWeight, Mathf.Pow(tt, 2))
                            : Mathf.Lerp(minWeight, maxWeight, tt);

                        totalWeight += weight;
                        if (item == 1)
                            count1 += Mathf.RoundToInt(weight);
                        ii++;
                    }
                    // Debug.Log("########################## totalWeight: " + totalWeight + ", count1:" + count1);
                    if (count1 >= totalWeight / 2f)
                    {
                        saber = saberL;
                        if (note_x == 1) correct_hit += 1;
                        else wrong_hit += 1;
                    }
                    else
                    {
                        saber = saberR;
                        if (note_x == 1) wrong_hit += 1;
                        else correct_hit += 1;
                    }

                }
                else correct_hit += 1;
            }

            Vector3 endPos = targetPos;
            float t = 0f;
            Vector3 startPos = saber.position;
            while (t < swingDuration)
            {
                saber.position = Vector3.Lerp(startPos, endPos, t / swingDuration);
                saber.rotation = Quaternion.Slerp(startRot, endRot, t / swingDuration);
                t += Time.deltaTime;
                yield return null;
            }
            saber.position = endPos; saber.rotation = endRot;
            // 往回砍
            t = 0f;
            while (t < swingDuration)
            {
                saber.position = Vector3.Lerp(endPos, startPos, t / swingDuration);
                saber.rotation = Quaternion.Slerp(endRot, startRot, t / swingDuration);
                t += Time.deltaTime;
                yield return null;
            }
            saber.position = startPos; saber.rotation = startRot;
        }
        
    }
    Quaternion GetSwingYaw(int direction)
    {
        switch (direction)
        {
            case 2: return Quaternion.Euler(0, -swingAngleY, -swingAngleZ / 2);  // 從左往右
            case 3: return Quaternion.Euler(0, swingAngleY, -swingAngleZ / 2);   // 從右往左
            case 1: return Quaternion.Euler(-swingAngleY, 0, -swingAngleZ / 2);    // 從上往下
            case 0: return Quaternion.Euler(swingAngleY, 0, -swingAngleZ / 2);  // 從下往上
            default: return Quaternion.Euler(swingAngleY, 0, -swingAngleZ / 2);   // 預設從上往下
        }
    }
    // 根據 direction 取得開始 offset（讓 saber 從砍擊方向來）
    Vector3 GetStartOffset(int direction)
    {
        switch (direction)
        {
            case 0: return Vector3.up * 0.5f;
            case 1: return Vector3.down * 0.5f;
            case 2: return Vector3.left * 0.5f;
            case 3: return Vector3.right * 0.5f;
            case 4: return (Vector3.up + Vector3.left).normalized * 0.5f;
            case 5: return (Vector3.up + Vector3.right).normalized * 0.5f;
            case 6: return (Vector3.down + Vector3.left).normalized * 0.5f;
            case 7: return (Vector3.down + Vector3.right).normalized * 0.5f;
            default: return Vector3.zero;
        }
    }

    // 根據 direction 給 Saber 一個旋轉方向
    Quaternion GetSwingRotation(int direction)
    {
        switch (direction)
        {
            case 0: return Quaternion.Euler(0, 0, 0);   // down
            case 1: return Quaternion.Euler(0, 0, 180); // up
            case 2: return Quaternion.Euler(0, 0, 90);  // left
            case 3: return Quaternion.Euler(0, 0, -90); // right
            case 4: return Quaternion.Euler(0, 0, 135); // down-right
            case 5: return Quaternion.Euler(0, 0, -135);// down-left
            case 6: return Quaternion.Euler(0, 0, 45);  // up-right
            case 7: return Quaternion.Euler(0, 0, -45); // up-left
            default: return Quaternion.identity;
        }
    }
}