using System.Collections.Generic;
using UnityEngine;

public class BinaryVisualizer : MonoBehaviour
{
     // 左右增長條，但是效果好像沒想像中的好
    [Header("Prefab and Anchor")]
    public GameObject particlePrefab;
    public Transform groundStrip;

    [Header("Settings")]
    public float spawnInterval = 0.1f;
    public float particleSpacing = 0.1f;
    public float fadeDuration = 1.0f;
    public int groupSize = 10;
    public int maxGroups = 1;

    private float timer = 0f;
    public int data_tmp = 0; // simulated input
    private Queue<int> buffer = new Queue<int>();

    private List<VisualParticle> leftParticles = new List<VisualParticle>();
    private List<VisualParticle> rightParticles = new List<VisualParticle>();

    private int leftIndex = 0;
    private int rightIndex = 0;

    void Start()
    {
        // Pre-instantiate particles
        int maxPerSide = groupSize * maxGroups;

        for (int i = 0; i < maxPerSide; i++)
        {
            GameObject l = Instantiate(particlePrefab, groundStrip.position, Quaternion.identity, transform);
            SetupParticle(l, Color.red, leftParticles);

            GameObject r = Instantiate(particlePrefab, groundStrip.position, Quaternion.identity, transform);
            SetupParticle(r, Color.blue, rightParticles);
        }
    }

    void SetupParticle(GameObject obj, Color color, List<VisualParticle> list)
    {
        MeshRenderer mr = obj.GetComponent<MeshRenderer>();
        Material mat = new Material(mr.sharedMaterial);
        mat.color = new Color(color.r, color.g, color.b, 0f); // initially transparent
        mr.material = mat;

        obj.SetActive(false);

        list.Add(new VisualParticle
        {
            gameObject = obj,
            renderer = mr,
            originalColor = color,
            timer = 0f,
            isActive = false
        });
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= spawnInterval)
        {
            timer -= spawnInterval;

            int data = data_tmp; // Replace with actual input
            buffer.Enqueue(data);
            if (buffer.Count > groupSize * maxGroups)
                buffer.Dequeue();

            Visualize();
        }

        UpdateFade(leftParticles);
        UpdateFade(rightParticles);
    }

    void Visualize()
    {
        int[] dataArray = buffer.ToArray();
        int count = dataArray.Length;

        leftIndex = 0;
        rightIndex = 0;

        for (int i = 0; i < count; i++)
        {
            int g = i / groupSize;
            float alpha = 1.0f - (float)g / maxGroups;

            if (dataArray[i] == 0)
                ShowParticle(leftParticles, ref leftIndex, -1, i, alpha);
            else
                ShowParticle(rightParticles, ref rightIndex, 1, i, alpha);
        }

        // Hide unused
        for (int i = leftIndex; i < leftParticles.Count; i++)
            leftParticles[i].gameObject.SetActive(false);

        for (int i = rightIndex; i < rightParticles.Count; i++)
            rightParticles[i].gameObject.SetActive(false);
    }

    void ShowParticle(List<VisualParticle> list, ref int index, int dir, int i, float alpha)
    {
        if (index >= list.Count) return;

        float groupOffset = ((list.Count - i - 1) % groupSize) * particleSpacing;  // list.Count - i - 1: 從中間發散， i : 會從旁邊到中間
        Vector3 pos = groundStrip.position + Vector3.right * dir * (groupOffset + particleSpacing);

        var p = list[list.Count-index-1];
        p.gameObject.SetActive(true);
        p.gameObject.transform.position = pos;
        p.renderer.material.color = new Color(p.originalColor.r, p.originalColor.g, p.originalColor.b, alpha);
        p.timer = 0f;
        p.isActive = true;

        index++;
    }

    void UpdateFade(List<VisualParticle> particles)
    {
        foreach (var p in particles)
        {
            if (!p.isActive || !p.gameObject.activeSelf) continue;

            p.timer += Time.deltaTime;
            float t = Mathf.Clamp01(p.timer / fadeDuration);
            Color c = p.renderer.material.color;
            c.a = Mathf.Lerp(1f, 0f, t);
            p.renderer.material.color = c;

            if (t >= 1f)
            {
                p.gameObject.SetActive(false);
                p.isActive = false;
            }
        }
    }

    class VisualParticle
    {
        public GameObject gameObject;
        public MeshRenderer renderer;
        public Color originalColor;
        public float timer;
        public bool isActive;
    }
}
// 下面是用生成和 destory 的方法顯示，但是感覺太 overhead 了
/*using System.Collections.Generic;
using UnityEngine;

public class BinaryVisualizer : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject particlePrefab;
    public int data_tmp = 0;
    [Header("Anchors")]
    //public Transform columnLeft;
    //public Transform columnRight;
    public Transform groundStrip;

    [Header("Settings")]
    public float spawnInterval = 0.1f;
    public float particleSpeed = 0.5f;
    public float fadeDuration = 1.0f;

    private Queue<int> buffer = new Queue<int>(); // buffer last 50 data points
    private float timer = 0f;

    private const int groupSize = 10;
    private const int maxGroups = 1;

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= spawnInterval)
        {
            timer -= spawnInterval;

            // int data = GetData(); // Replace with actual LSL data retrieval
            int data = data_tmp; // Replace with actual LSL data retrieval
            buffer.Enqueue(data);
            if (buffer.Count > groupSize * maxGroups)
                buffer.Dequeue();

            Visualize();
        }
    }

    int GetData()
    {
        // Replace with your LSL reader logic
        // int data = Random.value > 0.5f ? 1 : 0;
        // Debug.Log("@@@@@@@@@@@@@@@@@@@ data:" + data);
        // return data;
    }

void Visualize()
    {
        // Grouping
        int[] dataArray = new int[buffer.Count];
        buffer.CopyTo(dataArray, 0);
        int groupCount = dataArray.Length / groupSize;

        for (int g = 0; g < groupCount; g++)
        {
            int start = g * groupSize;
            int zeros = 0;
            for (int i = 0; i < groupSize; i++)
            {
                if (dataArray[start + i] == 0) zeros++;
            }

            int ones = groupSize - zeros;
            float fade = 1.0f - (float)g / maxGroups;

            SpawnGroup(zeros, ones, fade);
        }
    }

    void SpawnGroup(int zeros, int ones, float alpha)
    {
        float spacing = 0.1f;

        for (int i = 0; i < zeros; i++)  // 往左
        {
            Vector3 offset_left = Vector3.left * (i + 1) * spacing;
            //Vector3 offset_back = -columnRight.forward * (i + 1) * spacing;
            //CreateParticle(columnLeft.position + offset_back, Color.red, alpha, -columnLeft.forward);
            CreateParticle(groundStrip.position + offset_left, Color.red, alpha, Vector3.left); // 地上的往左右長
        }

        for (int i = 0; i < ones; i++)  // 往右
        {
            Vector3 offset_right = Vector3.right * (i + 1) * spacing;
            //Vector3 offset_back = -columnRight.forward * (i + 1) * spacing;
            //CreateParticle(columnRight.position + offset_back, Color.blue, alpha, -columnRight.forward);
            CreateParticle(groundStrip.position + offset_right, Color.blue, alpha, Vector3.right);
        }
    }

    void CreateParticle(Vector3 position, Color color, float alpha, Vector3 direction)
    {
        GameObject p = Instantiate(particlePrefab, position, Quaternion.identity);
        var renderer = p.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            Material matInstance = new Material(renderer.sharedMaterial); // Clone material
            color.a = alpha;
            matInstance.color = color;
            renderer.material = matInstance;
        }
        Destroy(p, fadeDuration);
        p.AddComponent<ParticleMover>().Init(direction.normalized * particleSpeed, fadeDuration);
    }
}
*/