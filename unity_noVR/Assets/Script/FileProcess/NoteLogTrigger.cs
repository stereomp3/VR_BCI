using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[System.Serializable]
public class NoteLogTrigger : MonoBehaviour
{
    public int label = 0; // red = 1 left hand, blue = 0 right hand // 20250925 modify this ...
    public float death_time = 1f;

    private GameManager GM;
    private bool is_in_startlog = false;
    private bool is_in_endlog = false;
    private Transform pp;
    private float beatThreshold_up;
    void Start()
    {
        GM = GameManager.instance;
        if(transform.parent) if(transform.parent.parent) pp = transform.parent.parent;
        beatThreshold_up = Config.cube_space_time * Config.group_note_num + 2;
    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnTriggerEnter(Collider other)
    {
        
        if (other.CompareTag("StartLog"))
        {
            if (is_in_startlog) return;
            if (GM.setLogCallback != null) GM.setLogCallback.Invoke(label, LogType.Spawn);
            Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@ StartLog");
            is_in_startlog = true;
        }
        if (other.CompareTag("EndLog"))
        {
            if (is_in_endlog) return;
            // if (GM.setLogCallback != null) GM.setLogCallback.Invoke(label, LogType.End);
            Debug.Log("@@@@@@@@@@@@@@@@@@@@@@@ Start end");
            if (pp != null)
            {
                StartCoroutine(ShrinkAndDestroy());
            }
            is_in_endlog = true;
        }
    }

    // scale、return end log, detory obj
    IEnumerator ShrinkAndDestroy()
    {
        float timer = 0f;
        // float shrinkDuration = 2f;
        float shrinkDuration = (pp.localScale.z / 26) * beatThreshold_up; // 26 為 long note prefab 原始長度
        Vector3 initialScale = pp.localScale;
        // Debug.Log("@@@@@@@@@@@@@@@@@@@@ shrinkDuration: " + shrinkDuration);
        // 在 n 秒內執行迴圈
        while (timer < shrinkDuration)
        {
            timer += Time.deltaTime;
            float progress = timer / shrinkDuration; // 0 到 1 的進度

            // 計算新的 Z 軸 (從原本的 Z 變到 0)
            float newZ = Mathf.Lerp(initialScale.z, 0f, progress);

            // 更新 Scale (保持 X 和 Y 不變，只改 Z)
            pp.localScale = new Vector3(initialScale.x, initialScale.y, newZ);

            yield return null; // 等待下一偵
        }
        
        // 確保最後 Z 軸真的是 0
        pp.localScale = new Vector3(initialScale.x, initialScale.y, 0f);
        StartCoroutine(WaitAndEnd(0.5f)); // 讓 log 晚點送出
    }

    //  final to send log
    void FinalizeEndLog()
    {
        if (GM.setLogCallback != null) GM.setLogCallback.Invoke(label, LogType.End);
        // destory obj
        if (pp != null) Destroy(pp.gameObject);
        else Destroy(gameObject); 
    }

    IEnumerator WaitAndEnd(float wait_time)
    {
        yield return new WaitForSeconds(wait_time); // 讓 log 晚點送出
        // 縮小完成，執行最後步驟
        FinalizeEndLog();
    }
}
