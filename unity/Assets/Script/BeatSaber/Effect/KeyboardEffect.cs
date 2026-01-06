using System.Collections;
using System.Collections.Generic;
using UnityEngine;
[RequireComponent(typeof(AudioSource))]
public class KeyboardEffect : MonoBehaviour
{
    [Header("Material Settings")]
    public Color normalColor = Color.gray;
    public Color highlightColor = Color.cyan;
    public float fadeDuration = 0.3f;
    
    [Header("Audio Settings")]
    public AudioClip hitSound;
    public float volume = 0.7f;

    public GameObject Track;
    private Renderer keyRenderer;
    private Material keyMaterial;
    private Coroutine fadeCoroutine;
    private AudioSource audioSource;

    void Start()
    {
        keyRenderer = Track.GetComponent<Renderer>();
        keyMaterial = keyRenderer.material; // Create a unique instance
        keyMaterial.color = normalColor;

        audioSource = GetComponent<AudioSource>();
        audioSource.playOnAwake = false;
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("saber"))
        {
            TriggerHighlight();
            
        }
    }

    void TriggerHighlight()
    {
        if (fadeCoroutine != null)
            StopCoroutine(fadeCoroutine);

        // Play sound
        if (hitSound != null)
            audioSource.PlayOneShot(hitSound, Config.volume);

        fadeCoroutine = StartCoroutine(FadeHighlight());
    }

    IEnumerator FadeHighlight()
    {
        float elapsed = 0f;

        // Step 1: Instant switch to highlight color
        keyMaterial.color = highlightColor;

        // Step 2: Smooth fade back to normalColor
        while (elapsed < fadeDuration)
        {
            float t = elapsed / fadeDuration;
            keyMaterial.color = Color.Lerp(highlightColor, normalColor, t);
            elapsed += Time.deltaTime;
            yield return null;
        }

        keyMaterial.color = normalColor;
        fadeCoroutine = null;
    }
}
