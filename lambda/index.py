# lambda/index.py
import json
import os
import requests
import re
from botocore.exceptions import ClientError

# FastAPI エンドポイントの URL
FASTAPI_URL = os.environ.get("FASTAPI_URL","https://3899-34-168-150-150.ngrok-free.app")
# エンドポイントのパス
INFERENCE_ENDPOINT = "/inference"  # FastAPI サーバー上の推論エンドポイント

def lambda_handler(event, context):
    try:
        print("Received event:", json.dumps(event))
        
        # Cognitoで認証されたユーザー情報を取得
        user_info = None
        if 'requestContext' in event and 'authorizer' in event['requestContext']:
            user_info = event['requestContext']['authorizer']['claims']
            print(f"Authenticated user: {user_info.get('email') or user_info.get('cognito:username')}")
        
        # リクエストボディの解析
        body = json.loads(event['body'])
        message = body['message']
        conversation_history = body.get('conversationHistory', [])
        
        print("Processing message:", message)
        
        # 会話履歴を使用
        messages = conversation_history.copy()
        
        # ユーザーメッセージを追加
        messages.append({
            "role": "user",
            "content": message
        })
        
        # FastAPIサーバーに送信するリクエストペイロード
        request_payload = {
            "messages": messages,
            "config": {
                "maxTokens": 512,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
        
        print(f"Calling FastAPI inference endpoint at {FASTAPI_URL}{INFERENCE_ENDPOINT}")
        
        # FastAPIサーバーに推論リクエストを送信
        response = requests.post(
            f"{FASTAPI_URL}{INFERENCE_ENDPOINT}",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            timeout=30  # タイムアウト設定
        )
        
        # レスポンスの検証
        if response.status_code != 200:
            raise Exception(f"FastAPI server returned error: {response.status_code}, {response.text}")
        
        # レスポンスを解析
        response_data = response.json()
        print("FastAPI response:", json.dumps(response_data, default=str))
        
        # アシスタントの応答を取得
        assistant_response = response_data.get("response")
        
        if not assistant_response:
            raise Exception("No response content from the FastAPI server")
        
        # アシスタントの応答を会話履歴に追加
        messages.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        # 成功レスポンスの返却
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": True,
                "response": assistant_response,
                "conversationHistory": messages
            })
        }
        
    except requests.exceptions.RequestException as req_error:
        print(f"Request error: {str(req_error)}")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": f"FastAPI server connection error: {str(req_error)}"
            })
        }
    except Exception as error:
        print("Error:", str(error))
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "success": False,
                "error": str(error)
            })
        }
