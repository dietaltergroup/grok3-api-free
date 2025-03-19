Grok API Service

## Project Overview

This project provides a Python-based Grok API service that uses OpenAI's format conversion to call the official Grok website for API processing.  **Important Note:** Your IP address must not be blocked.  Execution failures may be due to this issue.

## Quick Start

1.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables:**

    Create a `.env` file and configure the following environment variables:

    ```dotenv
    # Server Configuration
    PORT=3000

    # API Configuration
    API_KEY=sk-123456789

    # Conversation Configuration
    IS_TEMP_CONVERSATION=false
    IS_TEMP_GROK2=true
    GROK2_CONCURRENCY_LEVEL=2

    # Image Hosting Configuration
    TUMY_KEY=your_tumy_key  # Choose either TUMY_KEY or PICGO_KEY
    # PICGO_KEY=your_picgo_key

    # SSO Configuration
    IS_CUSTOM_SSO=false

    # Display Configuration
    ISSHOW_SEARCH_RESULTS=false
    SHOW_THINKING=true

    # SSO Token Configuration (Separate multiple tokens with commas)
    SSO=your_sso_token1,your_sso_token2,your_sso_token3
    ```

3.  **Start the Service:**

    ```bash
    python app.py
    ```

## Environment Variable Explanation

| Variable                 | Description                                                                                                                                      | Required during Build | Example       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------- | ------------- |
| `IS_TEMP_CONVERSATION`   | Enables temporary conversations.  If enabled, conversation history is not saved on the webpage.                                                      | (Optional, default: `false`) | `true`/`false` |
| `IS_TEMP_GROK2`          | Enables unlimited temporary accounts for grok2. If disabled, grok2-related models will use the quota of your own cookie account.                       | (Optional, default: `true`) | `true`/`false` |
| `GROK2_CONCURRENCY_LEVEL` | Concurrency control for grok2 temporary accounts.  Setting it too high may result in an IP ban.                                                     | (Optional, default: `2`)   | `2`           |
| `API_KEY`                | Custom authentication key.                                                                                                                       | (Optional, default: `sk-123456789`) | `sk-123456789` |
| `PICGO_KEY`              | PicGo image hosting key. Choose either this or `TUMY_KEY`.                                                                                          | Required for streaming image generation | `-`           |
| `TUMY_KEY`               | TUMY image hosting key. Choose either this or `PICGO_KEY`.                                                                                           | Required for streaming image generation | `-`           |
| `ISSHOW_SEARCH_RESULTS`  | Whether to display search results.                                                                                                                | (Optional, default: `false`) | `true`/`false` |
| `SSO`                    | Grok website SSO Cookie.  Multiple cookies can be set, separated by commas. The code automatically rotates and balances between different accounts. | (Required unless `IS_CUSTOM_SSO` is enabled) | `sso1,sso2`   |
| `PORT`                   | Service deployment port.                                                                                                                         | (Optional, default: `3000`) | `3000`        |
| `IS_CUSTOM_SSO`          | If you want to use your own custom account pool for rotation and balancing, instead of the project's built-in logic, enable this option.  If enabled, `API_KEY` needs to be set to the SSO cookie used for request authentication, and the `SSO` environment variable will be ignored. | (Optional, default: `false`) | `true`/`false` |
| `SHOW_THINKING`          | Whether to display the thinking process of the thinking model.                                                                                   | (Optional, default: `true`) | `true`/`false` |

## Features

Implemented features:

*   Supports text-to-image generation, using the `grok-2-imageGen` and `grok-3-imageGen` models.
*   Supports image recognition and image uploading for all models. Only the latest image in user messages is recognized and stored; historical images are replaced with placeholders.
*   Supports search functionality, using the `grok-2-search` or `grok-3-search` models, with an option to disable search results.
*   Supports deep search functionality, using the `grok-3-deepsearch` model.
*   Supports reasoning model functionality, using the `grok-3-reasoning` model.
*   Supports true streaming.  All of the above features can be called in streaming mode.
*   Supports multi-account rotation, configured in the environment variables.
*   grok2 uses a temporary account mechanism, theoretically allowing unlimited calls.  You can also use your own account's grok2.
*   Option to remove the thinking process of the thinking model.
*   Supports custom rotation and load balancing, independent of the project code.
*   Converted to OpenAI format.

## Available Models

*   `grok-2`
*   `grok-2-imageGen`
*   `grok-2-search`
*   `grok-3`
*   `grok-3-search`
*   `grok-3-imageGen`
*   `grok-3-deepsearch`
*   `grok-3-reasoning`

## Model Usage Quota Reference

*   `grok-2`, `grok-2-imageGen`, `grok-2-search` combined: 20 calls, refreshed every 2 hours.
*   `grok-3`, `grok-3-search`, `grok-3-imageGen` combined: 20 calls, refreshed every 2 hours.
*   `grok-3-deepsearch`: 10 calls, refreshed every 24 hours.
*   `grok-3-reasoning`: 10 calls, refreshed every 24 hours.

## How to Obtain the Cookie:

1.  Open the official Grok website.
2.  Copy the SSO cookie value and paste it into the `SSO` variable.  [Instructions for obtaining SSO cookie](link_to_sso_instructions_if_available - if there's a link in the original, put it here).  If not, remove this sentence.

## API Calls

*   Model List: `/v1/models`
*   Chat: `/v1/chat/completions`

## Additional Notes

*   To use the streaming image generation feature, you need to apply for an API Key from the tumy image hosting service.
*   Automatically removes the `think` process from historical messages.  If a historical message contains base64 image text (instead of being uploaded via file upload), it is automatically converted to an `[Image]` placeholder.

## Disclaimer

⚠️ This project is for learning and research purposes only. Please comply with the relevant terms of use.

## Acknowledgements

This project is based on [xLmiler/grok2api](https://github.com/xLmiler/grok2api).  Special thanks to the original author for their contribution.
