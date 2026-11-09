import builtins
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args

T = TypeVar("T", bound="HubMixin")


class HubMixin:


    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)


        self._save_pretrained(save_directory)


        if push_to_hub:
            if repo_id is None:
                repo_id = save_directory.name
            return self.push_to_hub(repo_id=repo_id, card_kwargs=card_kwargs, **push_to_hub_kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:

        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:

        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        card_kwargs: dict[str, Any] | None = None,
    ) -> str:

        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        if commit_message is None:
            if "Policy" in self.__class__.__name__:
                commit_message = "Upload policy"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            else:
                commit_message = f"Upload {self.__class__.__name__}"


        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, card_kwargs=card_kwargs)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )
