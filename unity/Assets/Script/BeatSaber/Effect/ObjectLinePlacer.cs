using UnityEngine;

public class ObjectLinePlacer : MonoBehaviour
{
    public GameObject targetObject;      // 用來取得 Z 長度
    public GameObject objectPrefab;      // 要複製的物件
    public int objectCount = 5;          // 使用者輸入的 N 值

    void Start()
    {
        if (targetObject == null || objectPrefab == null || objectCount < 2)
        {
            Debug.LogError("請設定 Target Object、Prefab 且 objectCount >= 2");
            return;
        }

        float zLength = GetZLength(targetObject);
        float spacing = zLength / (objectCount - 1);

        GameObject[] objects = new GameObject[objectCount];

        for (int i = 0; i < objectCount; i++)
        {
            float zPos = spacing * i - (targetObject.transform.localScale.z / targetObject.transform.localPosition.z)* targetObject.transform.parent.localScale.z;// scalez/posz*parentz
            // float yPos = (i % 2 == 0) ? 0.3f : 0.7f;
            float yPos = 0.3f;
            Vector3 position = targetObject.transform.position + new Vector3(0, yPos, zPos); 
            GameObject obj = Instantiate(objectPrefab, position, Quaternion.identity, this.transform);
            obj.name = "Obj_" + i;
            objects[i] = obj;

            /*// 從第 1 個物件開始畫 Line，前一個物件作為父物件
            if (i > 0)
            {
                GameObject lineObj = new GameObject("Line_" + (i - 1) + "_" + i);
                lineObj.transform.parent = objects[i - 1].transform;

                LineRenderer line = lineObj.AddComponent<LineRenderer>();
                line.material = new Material(Shader.Find("Sprites/Default"));
                line.startWidth = 0.05f;
                line.endWidth = 0.05f;
                line.positionCount = 2;
                line.useWorldSpace = true;
                line.SetPosition(0, objects[i - 1].transform.position);
                line.SetPosition(1, obj.transform.position);
            }*/
        }
    }

    float GetZLength(GameObject obj)
    {
        Renderer renderer = obj.GetComponent<Renderer>();
        if (renderer != null)
        {
            return renderer.bounds.size.z;
        }

        Collider col = obj.GetComponent<Collider>();
        if (col != null)
        {
            return col.bounds.size.z;
        }

        Debug.LogWarning("找不到 Renderer 或 Collider 來計算 Z 長度，預設使用 1");
        return 1f;
    }
}