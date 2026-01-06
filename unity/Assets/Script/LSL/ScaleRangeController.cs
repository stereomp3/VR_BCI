using System.Collections.Generic;
using UnityEngine;

public class ScaleRangeController : MonoBehaviour
{
    public Material RightSaberMaterial; // Assign the material using the Shader Graph
    public Material LeftSaberMaterial; // Assign the material using the Shader Graph
    // [Range(0f, 1f)]
    [Range(0f, .5f)]
    public float RightCurrentScale = 0f;
    // [Range(0f, 1f)]
    [Range(0f, .5f)]
    public float LeftCurrentScale = 0f;

    public float updateInterval = 0.1f; // LSL or input interval
    // Adjust speeds
    public float speedUp = 0.2f;   // Faster upward 上升速度 // 每秒上升 0.2，上升速度為下降兩倍
    public float speedDown = 0.1f; // Slower downward 自然下降速度
    private float timer = 0f;

    private TCP_Client tcp_predict;
    private int maxHistory = 10;
    
    private float RightTargetScale = 0f;
    private float LeftTargetScale = 0f;

    void Start()
    {
        tcp_predict = TCP_Client.instance;
        maxHistory = tcp_predict.historyLength;
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= updateInterval)
        {
            timer -= updateInterval;

            // Calculate target based on number of 1s
            int ones = 0;
            foreach (int val in tcp_predict.valueHistory)
                if (val == 1) ones++;

            RightTargetScale =  (maxHistory - ones) / (float)maxHistory; // 0 越多就把右手亮起來
            LeftTargetScale = ones / (float)maxHistory; // 1 越多就把左手量起來
        }
        // right stick
        if (RightCurrentScale < RightTargetScale)
        {
            RightCurrentScale = Mathf.MoveTowards(RightCurrentScale, RightTargetScale, speedUp * Time.deltaTime);
        }
        else if (RightCurrentScale > RightTargetScale)
        {
            RightCurrentScale = Mathf.MoveTowards(RightCurrentScale, RightTargetScale, speedDown * Time.deltaTime);
        }

        // Apply to material
        RightSaberMaterial.SetFloat("_scale_range", RightCurrentScale);  // 目前是設定 0.5 就是滿的，也就是 history 裡面有 5 個重複就可以滿
        // left stick
        if (LeftCurrentScale < LeftTargetScale)
        {
            LeftCurrentScale = Mathf.MoveTowards(LeftCurrentScale, LeftTargetScale, speedUp * Time.deltaTime);
        }
        else if (LeftCurrentScale > LeftTargetScale)
        {
            LeftCurrentScale = Mathf.MoveTowards(LeftCurrentScale, LeftTargetScale, speedDown * Time.deltaTime);
        }

        // Apply to material
        LeftSaberMaterial.SetFloat("_scale_range", LeftCurrentScale);  // 目前是設定 0.5 就是滿的，也就是 history 裡面有 5 個重複就可以滿
    }


}
