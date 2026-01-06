using UnityEngine;
using System.Collections.Generic;

public class LongNote : MonoBehaviour
{
    
    public forward forward;
    public Vector3 endPosition; // 終點偵測線
    private float speed; // 需要和 forward 一樣
    private bool isShrinking = false;
    private float currentShrinkIndex = 0f;
    private Vector3 originalScale;

    void Start()
    {
        originalScale = transform.localScale;
        speed = forward.speed;
    }

    void Update()
    {
        // AnimateWave();
        
        if (!isShrinking && IsTouchingEnd())
        {
            isShrinking = true;
        }

        if (isShrinking)
        {
            ShrinkLine();
        }
    }


    bool IsTouchingEnd()
    {
        Vector3 lastPoint = forward.transform.position;
        return Vector3.Distance(lastPoint, endPosition) < 0.5f;
    }

    void ShrinkLine()
    {
        // 縮小 localScale.z
        Vector3 newScale = transform.localScale;
        newScale.z -= Time.deltaTime * speed;

        // 限制不能小於 0
        if (newScale.z <= 0f)
        {
            newScale.z = 0f;
            isShrinking = false; // 可根據需求決定是否要銷毀
            forward.DestroyObjects();
        }

        transform.localScale = newScale;
    }
}