using UnityEngine;

public class ParticleMover : MonoBehaviour
{
    private Vector3 velocity;
    private float life;
    private float timer;
    private Material mat;
    private Color originalColor;

    public void Init(Vector3 vel, float duration)
    {
        velocity = vel; // move forward
        life = duration;

        MeshRenderer mr = GetComponent<MeshRenderer>();
        if (mr != null)
        {
            // Clone material so each particle fades independently
            mat = mr.material;
            originalColor = mat.color;
        }
    }

    void Update()
    {
        transform.position += velocity * Time.deltaTime;

        if (mat != null)
        {
            timer += Time.deltaTime;
            float alpha = Mathf.Lerp(originalColor.a, 0f, timer / life);
            Color c = originalColor;
            c.a = alpha;
            mat.color = c;
        }

        if (timer >= life)
        {
            Destroy(gameObject);
        }
    }
}