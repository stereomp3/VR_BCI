using UnityEngine;

public class AutoFistWithOVR : MonoBehaviour
{
    public Animator handAnimator; // 指向 OVRHandPrefab 裡的 Animator
    private bool isFist = false;

    void Start()
    {
        InvokeRepeating(nameof(ToggleFist), 0f, 0.2f);
    }

    void ToggleFist()
    {
        isFist = !isFist;
        handAnimator.SetFloat("Flex", isFist ? 1f : 0f);
        // Flex 是 OVRHandPrefab 內建的握拳驅動參數
    }
}