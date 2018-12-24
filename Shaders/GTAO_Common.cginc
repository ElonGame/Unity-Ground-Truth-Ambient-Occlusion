#include "UnityCG.cginc"
#include "Shader_Include/Include_HLSL.hlsl"

#define BLUR_RADIUS 8

int _AO_MultiBounce;
half _AO_DirSampler, _AO_SliceSampler, _AO_Intensity, _AO_Radius, _AO_Power, _AO_Sharpeness, _AO_TemporalScale, _AO_TemporalResponse, _AO_HalfProjScale, _AO_TemporalOffsets, _AO_TemporalDirections;
half2 _AO_FadeParams;
half4	_AO_UVToView, _AO_RT_TexelSize, _AO_FadeValues;
half4x4	_WorldToCameraMatrix, _CameraToWorldMatrix, _Inverse_View_ProjectionMatrix;
sampler2D _AO_Scene_Color, _CameraGBufferTexture0, _CameraGBufferTexture1, _CameraGBufferTexture2, _CameraReflectionsTexture, _CameraMotionVectorsTexture, _CameraDepthTexture, _CameraDepthNormalsTexture, _BentNormal_Texture, _GTAO_Texture, _GTAO_Spatial_Texture, _PrevRT, _CurrRT;

struct VertexInput
{
	half4 vertex : POSITION;
	half4 uv : TEXCOORD0;
};

struct PixelInput
{
	half4 vertex : SV_POSITION;
	half4 uv : TEXCOORD0;
};

PixelInput vert(VertexInput v)
{
	PixelInput o;
	o.vertex = v.vertex;
	o.uv = v.uv;
	return o;
}

//---//---//----//----//-------//----//----//----//-----//----//-----//----//----MultiBounce & ReflectionOcclusion//---//---//----//----//-------//----//----//----//-----//----//-----//----//----
inline float ConeConeIntersection(float ArcLength0, float ArcLength1, float AngleBetweenCones)
{
	float AngleDifference = abs(ArcLength0 - ArcLength1);
	return smoothstep(0, 1, 1 - saturate((AngleBetweenCones - AngleDifference) / (ArcLength0 + ArcLength1 - AngleDifference)));
}

inline half ReflectionOcclusion(half3 BentNormal, half3 ReflectionVector, half Roughness, half OcclusionStrength)
{
	half BentNormalLength = length(BentNormal);
	half ReflectionConeAngle = max(Roughness, 0.1) * PI;
	half UnoccludedAngle = BentNormalLength * PI * OcclusionStrength;
	half AngleBetween = acos(dot(BentNormal, ReflectionVector) / max(BentNormalLength, 0.001));

	half ReflectionOcclusion = ConeConeIntersection(ReflectionConeAngle, UnoccludedAngle, AngleBetween);
	return lerp(0, ReflectionOcclusion, saturate((UnoccludedAngle - 0.1) / 0.2));
}

inline half ReflectionOcclusion_Approch(half NoV, half Roughness, half AO)
{
	return saturate(pow(NoV + AO, Roughness * Roughness) - 1 + AO);
}

inline half3 MultiBounce(half AO, half3 Albedo)
{
	half3 A = 2 * Albedo - 0.33;
	half3 B = -4.8 * Albedo + 0.64;
	half3 C = 2.75 * Albedo + 0.69;
	return max(AO, ((AO * A + B) * AO + C) * AO);
}


//---//---//----//----//-------//----//----//----//-----//----//-----//----//----BilateralBlur//---//---//----//----//-------//----//----//----//-----//----//-----//----//----
inline void GetAo_Depth(float2 uv, inout float ao, inout float depth) {
	float2 aod = tex2Dlod(_GTAO_Texture, float4(uv, 0, 0)).rga;
	ao = aod.r;
	depth = aod.g;
}

inline float CrossBilateralWeight(float r, float d, float d0) {
	const float BlurSigma = (float)BLUR_RADIUS * 0.5;
	const float BlurFalloff = 1 / (2 * BlurSigma * BlurSigma);

    float dz = (d0 - d) * _ProjectionParams.z * _AO_Sharpeness;
	return exp2(-r * r * BlurFalloff - dz * dz);
}

inline void ProcessSample(float2 aoz, float r, float d0, inout float totalAO, inout float totalW) {
	float w = CrossBilateralWeight(r, d0, aoz.y);
	totalW += w;
	totalAO += w * aoz.x;
}

inline void ProcessRadius(float2 uv0, float2 deltaUV, float d0, inout float totalAO, inout float totalW) {
	float ao, z;
	float2 uv;
	float r = 1;

	UNITY_UNROLL
	for (; r <= BLUR_RADIUS / 2; r += 1) {
		uv = uv0 + r * deltaUV;
		GetAo_Depth(uv, ao, z);
		ProcessSample(float2(ao, z), r, d0, totalAO, totalW);
	}

	UNITY_UNROLL
	for (; r <= BLUR_RADIUS; r += 2) {
		uv = uv0 + (r + 0.5) * deltaUV;
		GetAo_Depth(uv, ao, z);
		ProcessSample(float2(ao, z), r, d0, totalAO, totalW);
	}
		
}

inline float2 BilateralBlur(float2 uv0, float2 deltaUV)
{
	float totalAO, depth;
	GetAo_Depth(uv0, totalAO, depth);
	float totalW = 1;
		
	ProcessRadius(uv0, -deltaUV, depth, totalAO, totalW);
	ProcessRadius(uv0, deltaUV, depth, totalAO, totalW);

	totalAO /= totalW;
	return float2(totalAO, depth);
}


//---//---//----//----//-------//----//----//----//-----//----//-----//----//----GTAO//---//---//----//----//-------//----//----//----//-----//----//-----//----//----
inline half ComputeDistanceFade(const half distance)
{
	return saturate(max(0, distance - _AO_FadeParams.x) * _AO_FadeParams.y);
}

inline half3 GetPosition(half2 uv)
{
	half depth = tex2Dlod(_CameraDepthTexture, float4(uv, 0, 0)).r; 
	half viewDepth = LinearEyeDepth(depth);
	return half3((uv * _AO_UVToView.xy + _AO_UVToView.zw) * viewDepth, viewDepth);
}

inline half3 GetNormal(half2 uv)
{
	half3 Normal = tex2D(_CameraGBufferTexture2, uv).rgb * 2 - 1; 
	half3 view_Normal = normalize(mul((half3x3) _WorldToCameraMatrix, Normal));

	return half3(view_Normal.xy, -view_Normal.z);
}

inline half GTAO_Offsets(half2 uv)
{
	int2 position = (int2)(uv * _AO_RT_TexelSize.zw);
	return 0.25 * (half)((position.y - position.x) & 3);
}

inline half GTAO_Noise(half2 position)
{
	return frac(52.9829189 * frac(dot(position, half2( 0.06711056, 0.00583715))));
}

half IntegrateArc_UniformWeight(half2 h)
{
	half2 Arc = 1 - cos(h);
	return Arc.x + Arc.y;
}

half IntegrateArc_CosWeight(half2 h, half n)
{
    half2 Arc = -cos(2 * h - n) + cos(n) + 2 * h * sin(n);
    return 0.25 * (Arc.x + Arc.y);
}

half4 GTAO(half2 uv, int NumCircle, int NumSlice, inout half Depth)
{
	half3 vPos = GetPosition(uv);
	half3 viewNormal = GetNormal(uv);
	half3 viewDir = normalize(0 - vPos);

	half2 radius_thickness = lerp(half2(_AO_Radius, 1), _AO_FadeValues.yw, ComputeDistanceFade(vPos.b).xx);
	half radius = radius_thickness.x;
	half thickness = radius_thickness.y;

	half stepRadius = (max(min((radius * _AO_HalfProjScale) / vPos.b, 512), (half)NumSlice)) / ((half)NumSlice + 1);
	half noiseOffset = frac(GTAO_Offsets(uv) + _AO_TemporalOffsets);
	half noiseDirection = GTAO_Noise(uv * _AO_RT_TexelSize.zw) + _AO_TemporalDirections;

	half Occlusion, angle,BentAngle, wallDarkeningCorrection, sliceLength, n, cos_n;
	half2 slideDir_TexelSize, h, H, falloff, uvOffset, h1h2, h1h2Length;
	half3 sliceDir, h1, h2, planeNormal, planeTangent, sliceNormal, BentNormal;
	half4 uvSlice;
	
	if (tex2D(_CameraDepthTexture, uv).r <= 1e-7) return 1;

	UNITY_LOOP
	for (int i = 0; i < NumCircle; i++)
	{
		angle = (i + noiseDirection) * (UNITY_PI / (half)NumCircle);
		sliceDir = half3(half2(cos(angle), sin(angle)), 0);

		planeNormal = normalize(cross(sliceDir, viewDir));
		planeTangent = cross(viewDir, planeNormal);
		sliceNormal = viewNormal - planeNormal * dot(viewNormal, planeNormal);
		sliceLength = length(sliceNormal);

		cos_n = clamp(dot(normalize(sliceNormal), viewDir), -1, 1);
		n = -sign(dot(sliceNormal, planeTangent)) * acos(cos_n);
		h = -1;

		UNITY_LOOP
		for (int j = 0; j < NumSlice; j++)
		{
			uvOffset = (sliceDir.xy * _AO_RT_TexelSize.xy) * max(stepRadius * (j + noiseOffset), 1 + j);
			uvSlice = uv.xyxy + float4(uvOffset.xy, -uvOffset);

			h1 = GetPosition(uvSlice.xy) - vPos;
			h2 = GetPosition(uvSlice.zw) - vPos;

			h1h2 = half2(dot(h1, h1), dot(h2, h2));
			h1h2Length = rsqrt(h1h2);

			falloff = saturate(h1h2 * (2 / pow2(radius)));

			H = half2(dot(h1, viewDir), dot(h2, viewDir)) * h1h2Length;
			h.xy = (H.xy > h.xy) ? lerp(H, h, falloff) : lerp(H.xy, h.xy, thickness);
		}

		h = acos(clamp(h, -1, 1));
		h.x = n + max(-h.x - n, -UNITY_HALF_PI);
		h.y = n + min(h.y - n, UNITY_HALF_PI);

		BentAngle = (h.x + h.y) * 0.5;
		BentNormal += viewDir * cos(BentAngle) - planeTangent * sin(BentAngle);

		Occlusion += sliceLength * IntegrateArc_CosWeight(h, n); 			
		//Occlusion += sliceLength * IntegrateArc_UniformWeight(h);			
	}

	BentNormal = normalize(normalize(BentNormal) - viewDir * 0.5);
	Occlusion = saturate(pow(Occlusion / (half)NumCircle, _AO_Power));
	Depth = vPos.b;

	return half4(BentNormal, Occlusion);
}
