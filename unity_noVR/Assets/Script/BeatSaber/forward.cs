using System;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class forward : MonoBehaviour
{
    public float speed = 10.0f;  // 結束的 speed，主要移動看下面
    public bool is_long_note = false;
    private bool is_stop = false;
    // 移動時間總共 1.5f
    private Vector3 targetPosition; // 終點位置 // z 
    private float duration = 1f; // 移動時間，單位為秒 2.5f
    private float animation_duration = 0.5f; // 移動時間，單位為秒 1.5f  

    private Vector3 startPosition;
    private float startTime;
    private bool is_arrive = false;
    private bool is_animation = true;
    private GameManager GM;
    // Start is called before the first frame update
    void Start()
    {
        GM = GameManager.instance;
        duration = GM.song_delay_time;
        targetPosition = new Vector3(transform.position.x, transform.position.y, 2f); // 這個 z 改變，連動需要改面 延遲時間、auto saber 揮砍時間
        startPosition = transform.position; // 記錄起始位置
        StartCoroutine(AnimateNote());
    }

    // Update is called once per frame
    void Update()
    {
        if (!is_long_note) // long note 不需要移動，只靠動畫移動到終點
        {
            if (!is_stop) // cube 的邏輯
            {
                if (is_arrive) transform.position += Vector3.back * Time.deltaTime * speed;
            }
            else
            {
                transform.position += Vector3.back * Time.deltaTime;
            }
            if (transform.position.z < -50)
            {
                DestroyObjects();
            }
        }
    }
    IEnumerator MoveToTarget()
    {
        while (Time.time - startTime < duration)
        {
            float t = (Time.time - startTime) / duration; // 計算插值比例，0到1之間
            transform.position = Vector3.Lerp(startPosition, targetPosition, t); // 線性插值
            yield return null; // 等待下一帧
        }

        transform.position = targetPosition; // 確保最終位置正確
        is_arrive = true;
    }

    IEnumerator AnimateNote()  // 衝出去動畫，要延遲可以把 Vector3.forward 乘小的值 (調整從後面多少衝到 spawn object)，然後 duration 上升
    {
        Vector3 start = startPosition + Vector3.forward * 100f;  // 300 f
        Vector3 end = startPosition;

        float t = 0;
        while (t < animation_duration)
        {
            t += Time.deltaTime;
            transform.position = Vector3.Lerp(start, end, t / animation_duration);
            yield return null;
        }
        transform.position = end;
        is_animation = false;

        startTime = Time.time; // 記錄開始時間
        duration = duration - animation_duration;
        StartCoroutine(MoveToTarget());
    }
    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("saber"))
        {
            if (!is_long_note)  is_stop = true;
            // Debug.Log("@@@@@@@@@@@@@@@@@@@@@Eeeeeeeee");
        }
    }

    public void DestroyObjects()
    {
        Destroy(gameObject);
    }
}
