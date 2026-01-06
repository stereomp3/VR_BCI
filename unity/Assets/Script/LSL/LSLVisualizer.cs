using System.Collections.Generic;
using UnityEngine;
using static Oculus.Interaction.Context;

public class LSLVisualizer : MonoBehaviour
{
    [Header("Target container positions")]
    public Transform leftLimit;      // Assign far left point in the Inspector
    public Transform rightLimit;     // Assign far right point in the Inspector

    [Header("Smoothing and Memory")]
    // public int historyLength = 10;   // How many recent values to average
    public float moveSpeed = 5f;     // Speed of interpolation

    // private ReceiveLSLMarker lSLMarker;
    private TCP_Client tcp_predict;
    void Start()
    {
        // lSLMarker = ReceiveLSLMarker.instance;
        tcp_predict = TCP_Client.instance;
    }

    void Update()
    {
        // Smooth movement between left (1) and right (0) based on average
        float t = Mathf.Lerp(1, 0, get_target_pos_nor(ref tcp_predict.valueHistory)); // use targetPositionNormalized 0~1 to set the pos
        Vector3 targetPos = Vector3.Lerp(leftLimit.position, rightLimit.position, t);  // (a, b, t), return a + (b - a) * t.
        transform.position = Vector3.Lerp(transform.position, targetPos, Time.deltaTime * moveSpeed);
    }

    
    public float get_target_pos_nor(ref Queue<int> valueHistory) // call this from your LSL script
    {
        // Calculate normalized average
        float sum = 0f;
        foreach (int v in valueHistory) sum += v;
        return sum / tcp_predict.historyLength; // 如果 historyLength 為 10 就是有 10 個刻度，根據目前輸入的情況移動
    }
}
