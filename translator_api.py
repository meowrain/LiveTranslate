"""Traditional translation API implementations.

Provides fast, low-latency translation via:
- Baidu Translate (百度翻译)
- Tencent Translate (腾讯翻译)
- Youdao Translate (有道翻译 / 网易)
- DeepL Translate
"""

import hashlib
import hmac
import json
import logging
import random
import time
from typing import Generator

import httpx

log = logging.getLogger("LiveTranslate.TL")

# ---------------------------------------------------------------------------
# Language code mappings per API
# ---------------------------------------------------------------------------

BAIDU_LANG = {
    "zh": "zh", "en": "en", "ja": "jp", "ko": "kor",
    "fr": "fra", "de": "de", "es": "spa", "ru": "ru",
    "pt": "pt", "it": "it", "nl": "nl", "pl": "pl",
    "tr": "tr", "ar": "ara", "th": "th", "vi": "vie",
    "id": "id", "ms": "may", "hi": "hi", "uk": "uk",
    "cs": "cs", "ro": "rom", "el": "el", "hu": "hu",
    "sv": "swe", "da": "dan", "fi": "fin", "no": "nor", "he": "iw",
}

TENCENT_LANG = {
    "zh": "zh", "en": "en", "ja": "ja", "ko": "ko",
    "fr": "fr", "de": "de", "es": "es", "ru": "ru",
    "pt": "pt", "it": "it", "nl": "nl", "pl": "pl",
    "tr": "tr", "ar": "ar", "th": "th", "vi": "vi",
    "id": "id", "ms": "ms", "hi": "hi", "uk": "uk",
    "cs": "cs", "ro": "ro", "el": "el", "hu": "hu",
    "sv": "sv", "da": "da", "fi": "fi", "no": "no", "he": "he",
}

YOUDAO_LANG = {
    "zh": "zh-CHS", "en": "en", "ja": "ja", "ko": "ko",
    "fr": "fr", "de": "de", "es": "es", "ru": "ru",
    "pt": "pt", "it": "it", "nl": "nl", "pl": "pl",
    "tr": "tr", "ar": "ar", "th": "th", "vi": "vi",
    "id": "id", "ms": "ms", "hi": "hi", "uk": "uk",
    "cs": "cs", "ro": "ro", "el": "el", "hu": "hu",
    "sv": "sv", "da": "da", "fi": "fi", "no": "no", "he": "he",
}

DEEPL_LANG = {
    "zh": "ZH", "en": "EN", "ja": "JA", "ko": "KO",
    "fr": "FR", "de": "DE", "es": "ES", "ru": "RU",
    "pt": "PT", "it": "IT", "nl": "NL", "pl": "PL",
    "tr": "TR", "ar": "AR", "th": "TH", "vi": "VI",
    "id": "ID", "ms": "MS", "hi": "HI", "uk": "UK",
    "cs": "CS", "ro": "RO", "el": "EL", "hu": "HU",
    "sv": "SV", "da": "DA", "fi": "FI", "no": "NB", "he": "HE",
}

# Translator type identifiers (used in user_settings.json model config "type" field)
TRANSLATOR_TYPES = {
    "llm": "AI (LLM)",
    "baidu": "Baidu",
    "tencent": "Tencent",
    "youdao": "Youdao",
    "deepl": "DeepL",
}


def _make_http_client(proxy: str = "none", timeout: int = 10) -> httpx.Client:
    """Create an httpx client respecting proxy settings (same logic as make_openai_client)."""
    if proxy == "system":
        return httpx.Client(timeout=timeout)
    elif proxy in ("none", "", None):
        return httpx.Client(trust_env=False, timeout=timeout)
    else:
        return httpx.Client(proxy=proxy, timeout=timeout)


# ---------------------------------------------------------------------------
# Baidu Translate  (百度翻译)
# ---------------------------------------------------------------------------

class BaiduTranslator:
    """Baidu Translate API.

    Required model config fields: app_id, secret_key
    API docs: https://fanyi-api.baidu.com/doc/21
    """

    def __init__(self, app_id: str, secret_key: str,
                 target_language: str = "zh", proxy: str = "none", timeout: int = 10):
        self._app_id = app_id
        self._secret_key = secret_key
        self._target_language = target_language
        self._proxy = proxy
        self._timeout = timeout
        self._client = _make_http_client(proxy, timeout)
        self._url = "https://fanyi-api.baidu.com/api/trans/vip/translate"

    def _sign(self, q: str, salt: str) -> str:
        raw = self._app_id + q + salt + self._secret_key
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def translate(self, text: str, source_language: str = "en") -> str:
        salt = str(random.randint(10000, 99999))
        sign = self._sign(text, salt)
        src = BAIDU_LANG.get(source_language, source_language)
        tgt = BAIDU_LANG.get(self._target_language, self._target_language)
        params = {
            "q": text, "from": src, "to": tgt,
            "appid": self._app_id, "salt": salt, "sign": sign,
        }
        resp = self._client.get(self._url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if "error_code" in data:
            raise RuntimeError(
                f"Baidu API error {data['error_code']}: {data.get('error_msg', '')}"
            )
        results = data.get("trans_result", [])
        return "\n".join(r["dst"] for r in results)

    def translate_iter(self, text: str, source_language: str = "en") -> Generator[str, None, None]:
        yield self.translate(text, source_language)

    def set_target_language(self, lang: str):
        self._target_language = lang

    def set_timeout(self, timeout: int):
        self._timeout = timeout
        self._client = _make_http_client(self._proxy, timeout)

    def set_context_turns(self, n: int):
        pass

    def clear_history(self):
        pass

    def with_target_language(self, target_language: str) -> "BaiduTranslator":
        t = BaiduTranslator.__new__(BaiduTranslator)
        t._app_id = self._app_id
        t._secret_key = self._secret_key
        t._target_language = target_language
        t._proxy = self._proxy
        t._timeout = self._timeout
        t._client = self._client
        t._url = self._url
        return t

    @property
    def last_usage(self):
        return (0, 0)


# ---------------------------------------------------------------------------
# Tencent Translate  (腾讯翻译)
# ---------------------------------------------------------------------------

class TencentTranslator:
    """Tencent Cloud Machine Translation API (TC3-HMAC-SHA256).

    Required model config fields: secret_id, secret_key, region (optional)
    API docs: https://cloud.tencent.com/document/api/551/15619
    """

    def __init__(self, secret_id: str, secret_key: str, region: str = "ap-guangzhou",
                 target_language: str = "zh", proxy: str = "none", timeout: int = 10):
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._region = region
        self._target_language = target_language
        self._proxy = proxy
        self._timeout = timeout
        self._client = _make_http_client(proxy, timeout)
        self._host = "tmt.tencentcloudapi.com"
        self._url = f"https://{self._host}"
        self._service = "tmt"
        self._action = "TextTranslate"
        self._version = "2018-03-21"

    def _build_auth(self, payload: str, timestamp: int) -> dict:
        date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
        payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # Step 1: Canonical request
        canonical_request = "\n".join([
            "POST", "/", "",
            "content-type:application/json",
            f"host:{self._host}",
            "",
            "content-type;host",
            payload_hash,
        ])

        # Step 2: String to sign
        credential_scope = f"{date}/{self._service}/tc3_request"
        string_to_sign = "\n".join([
            "TC3-HMAC-SHA256",
            str(timestamp),
            credential_scope,
            hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
        ])

        # Step 3: Signature
        def _hmac(key: bytes, msg: str) -> bytes:
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

        secret_date = _hmac(("TC3" + self._secret_key).encode("utf-8"), date)
        secret_service = _hmac(secret_date, self._service)
        secret_signing = _hmac(secret_service, "tc3_request")
        sig = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

        # Step 4: Authorization header
        auth = (
            f"TC3-HMAC-SHA256 Credential={self._secret_id}/{credential_scope}, "
            f"SignedHeaders=content-type;host, Signature={sig}"
        )
        return {
            "Authorization": auth,
            "Content-Type": "application/json",
            "Host": self._host,
            "X-TC-Action": self._action,
            "X-TC-Version": self._version,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Region": self._region,
        }

    def translate(self, text: str, source_language: str = "en") -> str:
        src = TENCENT_LANG.get(source_language, source_language)
        tgt = TENCENT_LANG.get(self._target_language, self._target_language)
        body = json.dumps({
            "SourceText": text, "Source": src, "Target": tgt, "ProjectId": 0,
        })
        timestamp = int(time.time())
        headers = self._build_auth(body, timestamp)
        resp = self._client.post(self._url, content=body, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if "Response" in data and "Error" in data["Response"]:
            err = data["Response"]["Error"]
            raise RuntimeError(f"Tencent API error {err['Code']}: {err['Message']}")
        return data.get("Response", {}).get("TargetText", "")

    def translate_iter(self, text: str, source_language: str = "en") -> Generator[str, None, None]:
        yield self.translate(text, source_language)

    def set_target_language(self, lang: str):
        self._target_language = lang

    def set_timeout(self, timeout: int):
        self._timeout = timeout
        self._client = _make_http_client(self._proxy, timeout)

    def set_context_turns(self, n: int):
        pass

    def clear_history(self):
        pass

    def with_target_language(self, target_language: str) -> "TencentTranslator":
        t = TencentTranslator.__new__(TencentTranslator)
        t._secret_id = self._secret_id
        t._secret_key = self._secret_key
        t._region = self._region
        t._target_language = target_language
        t._proxy = self._proxy
        t._timeout = self._timeout
        t._client = self._client
        t._host = self._host
        t._url = self._url
        t._service = self._service
        t._action = self._action
        t._version = self._version
        return t

    @property
    def last_usage(self):
        return (0, 0)


# ---------------------------------------------------------------------------
# Youdao Translate  (有道翻译 / 网易)
# ---------------------------------------------------------------------------

class YoudaoTranslator:
    """Youdao Translate API (有道智云).

    Required model config fields: app_key, app_secret
    API docs: https://ai.youdao.com/DOCSIRMA/html/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E7%BF%BB%E8%AF%91/API%E6%96%87%E6%A1%A3/%E6%96%87%E6%9C%AC%E7%BF%BB%E8%AF%91%E6%9C%8D%E5%8A%A1/%E6%96%87%E6%9C%AC%E7%BF%BB%E8%AF%91%E6%9C%8D%E5%8A%A1-API%E6%96%87%E6%A1%A3.html
    """

    def __init__(self, app_key: str, app_secret: str,
                 target_language: str = "zh", proxy: str = "none", timeout: int = 10):
        self._app_key = app_key
        self._app_secret = app_secret
        self._target_language = target_language
        self._proxy = proxy
        self._timeout = timeout
        self._client = _make_http_client(proxy, timeout)
        self._url = "https://openapi.youdao.com/api"

    @staticmethod
    def _truncate(q: str) -> str:
        """Youdao truncation rule: first 10 + len + last 10 when len > 20."""
        return q if len(q) <= 20 else q[:10] + str(len(q)) + q[-10:]

    def _sign(self, q: str, salt: str, curtime: str) -> str:
        input_text = self._truncate(q)
        sign_str = self._app_key + input_text + salt + curtime + self._app_secret
        return hashlib.sha256(sign_str.encode("utf-8")).hexdigest()

    def translate(self, text: str, source_language: str = "en") -> str:
        salt = str(random.randint(10000, 99999))
        curtime = str(int(time.time()))
        sign = self._sign(text, salt, curtime)
        src = YOUDAO_LANG.get(source_language, source_language)
        tgt = YOUDAO_LANG.get(self._target_language, self._target_language)
        form = {
            "q": text, "from": src, "to": tgt,
            "appKey": self._app_key, "salt": salt,
            "sign": sign, "signType": "v3", "curtime": curtime,
        }
        resp = self._client.post(self._url, data=form)
        resp.raise_for_status()
        data = resp.json()
        error_code = data.get("errorCode", "0")
        if error_code != "0":
            raise RuntimeError(f"Youdao API error: {error_code}")
        translations = data.get("translation", [])
        return "\n".join(translations) if translations else ""

    def translate_iter(self, text: str, source_language: str = "en") -> Generator[str, None, None]:
        yield self.translate(text, source_language)

    def set_target_language(self, lang: str):
        self._target_language = lang

    def set_timeout(self, timeout: int):
        self._timeout = timeout
        self._client = _make_http_client(self._proxy, timeout)

    def set_context_turns(self, n: int):
        pass

    def clear_history(self):
        pass

    def with_target_language(self, target_language: str) -> "YoudaoTranslator":
        t = YoudaoTranslator.__new__(YoudaoTranslator)
        t._app_key = self._app_key
        t._app_secret = self._app_secret
        t._target_language = target_language
        t._proxy = self._proxy
        t._timeout = self._timeout
        t._client = self._client
        t._url = self._url
        return t

    @property
    def last_usage(self):
        return (0, 0)


# ---------------------------------------------------------------------------
# DeepL Translate
# ---------------------------------------------------------------------------

class DeepLTranslator:
    """DeepL Translate API.

    Required model config fields: api_key
    API docs: https://developers.deepl.com/docs/api-reference/translate/openapi-spec-for-text-translation
    """

    def __init__(self, api_key: str,
                 target_language: str = "zh", proxy: str = "none", timeout: int = 10):
        self._api_key = api_key
        self._target_language = target_language
        self._proxy = proxy
        self._timeout = timeout
        self._client = _make_http_client(proxy, timeout)
        # Free keys end with ":fx"
        if api_key.endswith(":fx"):
            self._url = "https://api-free.deepl.com/v2/translate"
        else:
            self._url = "https://api.deepl.com/v2/translate"

    def translate(self, text: str, source_language: str = "en") -> str:
        tgt = DEEPL_LANG.get(self._target_language, self._target_language).upper()
        form = {"text": [text], "target_lang": tgt}
        # DeepL auto-detects source; only set if explicitly provided and not "auto"
        if source_language and source_language != "auto":
            src = DEEPL_LANG.get(source_language, source_language).upper()
            form["source_lang"] = src
        headers = {"Authorization": f"DeepL-Auth-Key {self._api_key}"}
        resp = self._client.post(self._url, data=form, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        translations = data.get("translations", [])
        return translations[0]["text"] if translations else ""

    def translate_iter(self, text: str, source_language: str = "en") -> Generator[str, None, None]:
        yield self.translate(text, source_language)

    def set_target_language(self, lang: str):
        self._target_language = lang

    def set_timeout(self, timeout: int):
        self._timeout = timeout
        self._client = _make_http_client(self._proxy, timeout)

    def set_context_turns(self, n: int):
        pass

    def clear_history(self):
        pass

    def with_target_language(self, target_language: str) -> "DeepLTranslator":
        t = DeepLTranslator.__new__(DeepLTranslator)
        t._api_key = self._api_key
        t._target_language = target_language
        t._proxy = self._proxy
        t._timeout = self._timeout
        t._client = self._client
        t._url = self._url
        return t

    @property
    def last_usage(self):
        return (0, 0)
