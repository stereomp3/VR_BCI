using UnityEngine;

public class RippleTrigger : MonoBehaviour
{
    public ParticleSystem rippleParticle;
    public float particleSize = 5f;
    public float particleLifetime = 0.1f;
    public float forward = 1f;
    public Color particleColor = Color.white;
    private bool hasEmitted = false;

    private void OnTriggerEnter(Collider other)
    {
        if (rippleParticle == null) return;
        if (other.tag == "saber")
        {
            Vector3 closestPoint = other.ClosestPoint(transform.position);
            rippleParticle.transform.position = closestPoint + Vector3.down * 10 + Vector3.forward * forward;
            // 啟用粒子系統（不播放 loop）
            rippleParticle.gameObject.SetActive(true);
            if (!hasEmitted)
            {
                EmitAtPosition(closestPoint);
                hasEmitted = true;
            }
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (rippleParticle == null) return;

        rippleParticle.gameObject.SetActive(false);
        hasEmitted = false; // 重設可再次觸發
    }

    private void EmitAtPosition(Vector3 position)
    {
        /*var emitParams = new ParticleSystem.EmitParams();
        emitParams.position = position;
        emitParams.velocity = Vector3.zero;
        emitParams.startSize = particleSize;
        emitParams.startLifetime = particleLifetime;
        emitParams.startColor = particleColor;

        rippleParticle.Emit(emitParams, 1); // 發射一個粒子*/
        rippleParticle.Emit(1);
    }
}