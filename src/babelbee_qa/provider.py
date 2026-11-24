"""Small client for a user-operated OpenAI-compatible Chat Completions endpoint."""
from dataclasses import dataclass, field
import ipaddress
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .inputs import QAError

_MAX_RESPONSE = 2 * 1024 * 1024


class RetryableModelError(QAError):
    """A request may succeed if the same bounded input is sent again."""



def _loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == 'localhost'


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class Provider:
    base_url: str
    model: str
    key: str = field(default='', repr=False)
    timeout: int = 180

    def __post_init__(self):
        try:
            url = urlsplit(self.base_url)
            host = url.hostname
            port = url.port
        except ValueError as exc:
            raise QAError('Enter a valid model API base URL.') from exc
        if not host or url.username or url.password or url.query or url.fragment or port == 0:
            raise QAError('Use a model API base URL without credentials, query parameters or fragments.')
        is_loopback = _loopback(host)
        if url.scheme != 'https' and (url.scheme != 'http' or not is_loopback):
            raise QAError('The model endpoint must use HTTPS. Local loopback HTTP is also supported.')
        if not self.key and not is_loopback:
            raise QAError('Set BABELBEE_QA_API_KEY before calling a remote model.')
        if any(ord(char) < 33 or ord(char) > 126 for char in self.key):
            raise QAError('The model key must be a single printable token.')
        if not self.model.strip() or not 1 <= self.timeout <= 900:
            raise QAError('Set a model name and a timeout between 1 and 900 seconds.')

    def evaluate(self, prompt, payload):
        body = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))},
            ],
        }
        if (payload.get('task') == 'document_overview' and not payload.get('validation_retry')
                and self.model == 'gpt-6-luna'
                and urlsplit(self.base_url).hostname == 'api.openai.com'):
            body['reasoning_effort'] = 'none'
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.key:
            headers['Authorization'] = f'Bearer {self.key}'
        request = Request(self.base_url.rstrip('/') + '/chat/completions',
                          data=json.dumps(body, ensure_ascii=False).encode(), headers=headers)
        handlers = [_NoRedirect()]
        if _loopback(urlsplit(self.base_url).hostname):
            handlers.append(ProxyHandler({}))
        try:
            with build_opener(*handlers).open(request, timeout=self.timeout) as response:
                raw = response.read(_MAX_RESPONSE + 1)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            error = RetryableModelError if status in {408, 429, 500, 502, 503, 504} else QAError
            raise error(f'Model API returned HTTP {status}. Verify the endpoint, model, API key and limits.') from None
        except (URLError, OSError, ValueError):
            raise RetryableModelError('Cannot reach the model API or it took too long to respond.') from None
        if len(raw) > _MAX_RESPONSE:
            raise QAError('Model response exceeded the size limit.')
        try:
            choice = json.loads(raw)['choices'][0]
            if choice.get('finish_reason') not in {None, 'stop'} or choice['message'].get('refusal'):
                raise QAError('The model did not finish the review. Check its output and context limits.')
            content = choice['message']['content'].strip()
            if content.startswith('```') and content.endswith('```'):
                content = content.split('\n', 1)[-1][:-3].strip()
            data = json.loads(content)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise RetryableModelError('The model did not return valid review JSON. Try a model with reliable JSON output.') from None
        if not isinstance(data, dict):
            raise RetryableModelError('The model did not return a JSON object.')
        return data
