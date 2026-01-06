using UnityEngine;

public class HandFistChecker : MonoBehaviour
{
    public OVRHand left_hand; // 在 Inspector 指定 LeftHand 或 RightHand
    public OVRHand right_hand; // 在 Inspector 指定 LeftHand 或 RightHand

    void Update()
    {
        // Confidence 必須足夠，代表手部追蹤穩定
        if (right_hand.IsTracked && right_hand.HandConfidence == OVRHand.TrackingConfidence.High)
        {
            // Meta SDK 內建的手勢判斷：Fist
            if (right_hand.GetFingerIsPinching(OVRHand.HandFinger.Index) &&
                right_hand.GetFingerIsPinching(OVRHand.HandFinger.Middle) &&
                right_hand.GetFingerIsPinching(OVRHand.HandFinger.Ring) &&
                right_hand.GetFingerIsPinching(OVRHand.HandFinger.Pinky))
            {
                Debug.Log($"正在握拳 right hand");
            }
            //float gripStrength = hand.GetFingerPinchStrength(OVRHand.HandFinger.Middle);
            //if (gripStrength > 0.8f) { Debug.Log("正在握拳"); }
        }

        // Confidence 必須足夠，代表手部追蹤穩定
        if (left_hand.IsTracked && left_hand.HandConfidence == OVRHand.TrackingConfidence.High)
        {
            // Meta SDK 內建的手勢判斷：Fist
            if (left_hand.GetFingerIsPinching(OVRHand.HandFinger.Index) &&
                left_hand.GetFingerIsPinching(OVRHand.HandFinger.Middle) &&
                left_hand.GetFingerIsPinching(OVRHand.HandFinger.Ring) &&
                left_hand.GetFingerIsPinching(OVRHand.HandFinger.Pinky))
            {
                Debug.Log($"正在握拳 left hand");
            }
        }
    }
}