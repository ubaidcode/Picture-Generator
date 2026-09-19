import streamlit as st
from google import genai
from huggingface_hub import InferenceClient
from io import BytesIO


# ============================================================
# MAIN CONFIGURATION
# ============================================================

# 1. GEMINI API KEY
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# 2. GEMINI MODEL
GEMINI_MODEL = "gemini-3.5-flash"


# 3. HUGGING FACE TOKEN
HF_TOKEN = st.secrets["HF_TOKEN"]


# 4. HUGGING FACE IMAGE MODEL
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #777;
        margin-bottom: 35px;
    }

    .stButton > button {
        width: 100%;
        height: 55px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONFIGURATION CHECK
# ============================================================

if (
    not GEMINI_API_KEY
    or GEMINI_API_KEY == "PASTE_YOUR_GEMINI_API_KEY_HERE"
):
    st.error("Gemini API key is not configured.")
    st.info(
        "Open app.py and add your Gemini API key "
        "at the top of the file."
    )
    st.stop()


if (
    not HF_TOKEN
    or HF_TOKEN == "PASTE_YOUR_HUGGINGFACE_TOKEN_HERE"
):
    st.error("Hugging Face token is not configured.")
    st.info(
        "Open app.py and add your Hugging Face token "
        "at the top of the file."
    )
    st.stop()


# ============================================================
# API CLIENTS
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# IMPORTANT:
# No fixed Hugging Face provider is selected.
# The Hugging Face library will automatically select
# a supported provider for the selected model.

hf_client = InferenceClient(
    api_key=HF_TOKEN
)


# ============================================================
# GEMINI PROMPT ENHANCER
# ============================================================

def enhance_prompt(user_prompt):

    system_instruction = """
You are an expert professional AI image prompt engineer.

Transform the user's simple idea into a highly detailed,
professional prompt for an AI image generation model.

Improve the prompt with relevant details such as:

- Main subject
- Appearance
- Clothing
- Pose
- Facial expression
- Environment
- Background
- Lighting
- Camera angle
- Camera lens
- Composition
- Depth of field
- Realistic textures
- Colors
- Atmosphere
- Photorealistic details
- Image quality

Preserve the user's original idea and intent.

Do not change the main subject.

Do not add unrelated objects.

Do not explain anything.

Return ONLY the final image-generation prompt.
"""

    complete_prompt = (
        system_instruction
        + "\n\nUSER IDEA:\n"
        + user_prompt
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=complete_prompt
    )

    if not response.text:
        raise Exception(
            "Gemini did not return an enhanced prompt."
        )

    return response.text.strip()


# ============================================================
# HUGGING FACE IMAGE GENERATOR
# ============================================================

def generate_image(prompt):

    image = hf_client.text_to_image(
        prompt=prompt,
        model=HF_MODEL
    )

    if image is None:
        raise Exception(
            "Hugging Face did not return an image."
        )

    return image


# ============================================================
# CONVERT IMAGE TO PNG BYTES
# ============================================================

def image_to_bytes(image):

    image_buffer = BytesIO()

    image.save(
        image_buffer,
        format="PNG"
    )

    image_buffer.seek(0)

    return image_buffer.getvalue()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎨 AI Image Generator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Turn your ideas into professional AI-generated images'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# PROMPT INPUT
# ============================================================

st.subheader("Create Your Image")

user_prompt = st.text_area(
    "Enter your image idea",
    placeholder=(
        "Example: A young Pakistani man standing beside "
        "a motorcycle on a Karachi street during sunset..."
    ),
    height=160
)


# ============================================================
# GENERATE BUTTON
# ============================================================

generate_button = st.button(
    "✨ Generate Image",
    use_container_width=True
)


# ============================================================
# GENERATION PROCESS
# ============================================================

if generate_button:

    if not user_prompt.strip():
        st.warning(
            "Please enter an image idea first."
        )
        st.stop()

    try:

        # ====================================================
        # STEP 1 — ENHANCE PROMPT
        # ====================================================

        with st.status(
            "Improving your prompt...",
            expanded=True
        ) as status:

            st.write(
                f"Using Gemini model: {GEMINI_MODEL}"
            )

            enhanced_prompt = enhance_prompt(
                user_prompt.strip()
            )

            status.update(
                label="Professional prompt created",
                state="complete"
            )


        # ====================================================
        # SHOW ENHANCED PROMPT
        # ====================================================

        with st.expander(
            "View Enhanced Prompt"
        ):

            st.code(
                enhanced_prompt,
                language="text"
            )


        # ====================================================
        # STEP 2 — GENERATE IMAGE
        # ====================================================

        with st.status(
            "Generating your image...",
            expanded=True
        ) as status:

            st.write(
                f"Using Hugging Face model: {HF_MODEL}"
            )

            st.write(
                "Automatically selecting a supported "
                "Hugging Face provider..."
            )

            generated_image = generate_image(
                enhanced_prompt
            )

            status.update(
                label="Image generated successfully",
                state="complete"
            )


        # ====================================================
        # STEP 3 — CONVERT IMAGE
        # ====================================================

        image_bytes = image_to_bytes(
            generated_image
        )


        # ====================================================
        # STEP 4 — DISPLAY IMAGE
        # ====================================================

        st.success(
            "✅ Your image has been generated!"
        )

        st.image(
            generated_image,
            caption="AI Generated Image",
            use_container_width=True
        )


        # ====================================================
        # STEP 5 — DOWNLOAD
        # ====================================================

        st.download_button(
            label="⬇️ Download Image",
            data=image_bytes,
            file_name="ai_generated_image.png",
            mime="image/png",
            use_container_width=True
        )


    except Exception as error:

        st.error(
            "Image generation failed."
        )

        with st.expander(
            "Technical Error"
        ):

            st.code(
                str(error),
                language="text"
            )
