import json
from DrissionPage import ChromiumPage

# Initialize the browser page
driver = ChromiumPage()

# Start listening for the specific request
driver.listen.start('aweme/v1/web/comment/list/')

# Visit the Douyin video page
driver.get('https://www.douyin.com/video/7425089352549666082')

# Open the file for writing
with open("comments_output.txt", "w", encoding="utf-8") as file:
    comment_count = 0  # Initialize comment counter

    for page in range(10):  # Limit to 10 pages or until 30 comments are reached
        if comment_count >= 30:
            break  # Stop if we have collected 30 comments

        print(f'Collecting comments from page {page + 1}')
        driver.scroll.to_bottom()

        # Wait for the response of the comment request
        resp = driver.listen.wait()

        # Extract JSON data
        json_data = resp.response.body
        comments = json_data.get('comments', [])

        # Write comments to the file
        for index in comments:
            if comment_count >= 30:
                break  # Stop if 30 comments have been collected

            text = index.get('text', '')
            file.write(f"<comment>{text}</comment>\n")
            print(f"<comment>{text}</comment>")  # Optional: Print to console to track progress

            comment_count += 1  # Increment comment counter