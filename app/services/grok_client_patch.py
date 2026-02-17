    @staticmethod
    async def _collect_stream_to_text(response, session):
        """收集流式响应为完整文本"""
        try:
            content = ""
            grok_response_id = None

            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = orjson.loads(line)

                    if result := data.get("result"):
                        if response_data := result.get("response"):
                            if resp_id := response_data.get("responseId"):
                                grok_response_id = resp_id
                            elif model_resp := response_data.get("modelResponse"):
                                if resp_id := model_resp.get("responseId"):
                                    grok_response_id = resp_id

                            if token_text := response_data.get("token"):
                                if isinstance(token_text, str):
                                    content += token_text

                            if model_resp := response_data.get("modelResponse"):
                                if msg := model_resp.get("message"):
                                    content = msg

                except Exception as e:
                    logger.debug(f"[GrokClient] 流式解析失败: {e}")
                    continue

            logger.info(f"[GrokClient] 流式收集完成: resp_id={grok_response_id}, content_len={len(content)}")
            return content, grok_response_id

        finally:
            await session.close()
