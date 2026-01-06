Shader "Custom/TopBlackBottomWhite"
{
    Properties
    {
        _scale_range("Black Height Ratio (0=All Black, 1=All White)", Range(0, 1)) = 0.5
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
            };

            struct v2f
            {
                float4 pos : SV_POSITION;
                float3 worldPos : TEXCOORD0;
            };

            float _scale_range;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = TransformObjectToHClip(v.vertex.xyz);
                o.worldPos = TransformObjectToWorld(v.vertex.xyz);
                return o;
            }

            half4 frag (v2f i) : SV_Target
            {
                float y = i.worldPos.y;
                float topY = 0.5; // Adjust based on your object's actual height center
                float heightNorm = saturate((y + topY) / (2.0 * topY)); // Normalize Y 0~1

                float threshold = 1.0 - _scale_range;
                float blackMask = step(threshold, heightNorm); // 1 = black, 0 = white

                return lerp(1, 0, blackMask); // white = 1, black = 0
            }
            ENDHLSL
        }
    }
}