using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AudioEmissionController : MonoBehaviour
{
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private Renderer targetRenderer;
    [SerializeField] private Color emissiveColor = Color.white;
    [SerializeField, Range(1, 500)] private float sensitivity = 300f;
    [SerializeField, Range(0, 1)] private float smoothing = 0.5f;
    [SerializeField] private float threshold = 0.0001f;
    [SerializeField] private string emissionPropertyName = "_EmissionColor";
    [SerializeField] private float defaultIntensity = 0f;

    private Material material;
    private int emissionPropertyID;
    private float currentIntensity;

    private void Awake()
    {
        if (targetRenderer == null || audioSource == null)
        {
            if(targetRenderer) Debug.LogError("AudioSource not assigned." + gameObject.name);
            else Debug.LogError("Renderer not assigned.");
            enabled = false;
            return;
        }

        material = targetRenderer.material;  // Creates an instance at runtime (safe)
        emissionPropertyID = Shader.PropertyToID(emissionPropertyName);

        // Ensure emission is enabled
        if (!material.IsKeywordEnabled("_EMISSION"))
        {
            material.EnableKeyword("_EMISSION");
        }

        // Set a default emission value
        material.SetColor(emissionPropertyID, emissiveColor * defaultIntensity);
    }

    private void Update()
    {
        float[] spectrumData = new float[1024];
        audioSource.GetSpectrumData(spectrumData, 0, FFTWindow.BlackmanHarris);

        float audioAverage = 0f;
        foreach (float sample in spectrumData)
        {
            audioAverage += sample;
        }
        audioAverage /= spectrumData.Length;

        float targetIntensity = (audioAverage > threshold) ? audioAverage * sensitivity : defaultIntensity;
        currentIntensity = Mathf.Lerp(currentIntensity, targetIntensity, smoothing);

        material.SetColor(emissionPropertyID, emissiveColor * currentIntensity);
        /*Debug.Log("Intensity: audioAverage: " + audioAverage);
        Debug.Log("Intensity: targetIntensity: " + targetIntensity);
        Debug.Log("Intensity: currentIntensity: " + currentIntensity);
        Debug.Log("Intensity: audioSource: " + audioSource.isPlaying);*/
    }

    private void OnDestroy()
    {
        // Don't destroy the material unless you instantiated it manually with new Material()
        // Unity handles auto-created copies safely on its own.
    }
}