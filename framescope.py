from google import genai

client = genai.Client(api_key="GOOGLE_API_KEY")


def process_frame(frame, prompt):
    my_file = client.files.upload(file=frame)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[my_file, prompt],
    )

    return response.text