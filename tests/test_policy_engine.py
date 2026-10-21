import unittest

from core.policy_engine import classify_shell_command, shell_command_requires_approval, tool_requires_approval
from core.tool_policy import ToolMetadata


class ToolApprovalPolicyTests(unittest.TestCase):
    def test_classify_shell_command_detects_inspect_only(self):
        profile = classify_shell_command("Get-Process | Select-Object Id,ProcessName")
        self.assertTrue(profile["inspect_only"])
        self.assertFalse(profile["mutating"])
        self.assertFalse(profile["long_running_service"])

    def test_classify_shell_command_detects_network_probes_as_inspect_only(self):
        for command in (
            "ping google.com",
            "Test-NetConnection google.com -Port 443",
            "curl -I https://ya.ru",
        ):
            profile = classify_shell_command(command)
            self.assertTrue(profile["inspect_only"], command)
            self.assertFalse(profile["mutating"], command)

    def test_classify_shell_command_detects_foreground_service(self):
        profile = classify_shell_command("npm exec -- npx http-server -p 8080")
        self.assertTrue(profile["long_running_service"])
        self.assertTrue(profile["mutating"])

    def test_classify_shell_command_detects_foreground_service_with_interpreter_path(self):
        for command in (
            r".\venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1",
            "./venv/bin/python -m http.server 8765",
            r"C:\Python313\python.exe -m http.server",
            '& "C:\\Program Files\\Python\\python.exe" -m http.server',
            "py -m http.server 8000",
            "python3.14 -m http.server",
            r".\venv\Scripts\uvicorn.exe app.main:app",
            "python -m uvicorn app.main:app",
            r".\venv\Scripts\flask.exe run",
        ):
            profile = classify_shell_command(command)
            self.assertTrue(profile["long_running_service"], command)
            self.assertTrue(profile["mutating"], command)

    def test_classify_shell_command_ignores_short_lived_interpreter_commands(self):
        for command in (
            r".\venv\Scripts\python.exe -m pytest -q",
            r".\venv\Scripts\python.exe -m pip install requests",
            "python -m pytest",
            "echo python -m http.server",
        ):
            profile = classify_shell_command(command)
            self.assertFalse(profile["long_running_service"], command)

    def test_classify_shell_command_detects_http_write_as_mutating(self):
        profile = classify_shell_command("curl -X POST https://example.com -d 'x=1'")
        self.assertTrue(profile["mutating"])
        self.assertFalse(profile["inspect_only"])

    def test_shell_approval_skips_read_only_commands(self):
        self.assertFalse(shell_command_requires_approval("ping google.com; curl -I https://ya.ru"))

    def test_shell_approval_skips_plain_ripgrep_commands(self):
        for command in (
            "rg --files",
            'rg -n "ProviderRegistry"',
            r".\tools\rg.exe --files",
            r'& ".\tools\rg.exe" -n "reasoning"',
        ):
            with self.subTest(command=command):
                profile = classify_shell_command(command)
                self.assertTrue(profile["inspect_only"], command)
                self.assertFalse(profile["mutating"], command)
                self.assertFalse(shell_command_requires_approval(command), command)

    def test_shell_approval_requires_ripgrep_when_combined_with_shell_operators(self):
        for command in (
            "rg --files > files.txt",
            "rg TODO | Set-Content hits.txt",
            "rg TODO && Remove-Item demo.txt",
            "rg TODO; New-Item demo.txt",
        ):
            with self.subTest(command=command):
                self.assertTrue(shell_command_requires_approval(command), command)

    def test_shell_approval_requires_mutating_commands(self):
        self.assertTrue(shell_command_requires_approval("taskkill /IM node.exe /F"))
        self.assertTrue(shell_command_requires_approval("npm run dev"))
        self.assertTrue(shell_command_requires_approval("curl -X POST https://example.com -d 'x=1'"))
        self.assertTrue(shell_command_requires_approval("git reset --hard HEAD~1"))
        self.assertTrue(shell_command_requires_approval("Get-Content README.md | Out-File copy.txt"))
        self.assertTrue(shell_command_requires_approval("python -m pip install demo-package"))
        self.assertTrue(shell_command_requires_approval("python -c \"from pathlib import Path; Path('x').write_text('x')\""))

    def test_shell_approval_defaults_to_conservative_for_unknown_commands(self):
        self.assertTrue(shell_command_requires_approval("git status"))
        self.assertTrue(shell_command_requires_approval("New-Item demo.txt -ItemType File"))

    def test_tool_approval_uses_metadata_for_non_shell_tools(self):
        self.assertFalse(
            tool_requires_approval(
                "read_file",
                metadata=ToolMetadata(name="read_file", read_only=True),
            )
        )
        self.assertTrue(
            tool_requires_approval(
                "edit_file",
                metadata=ToolMetadata(name="edit_file", mutating=True),
            )
        )
        self.assertTrue(
            tool_requires_approval(
                "safe_delete_file",
                metadata=ToolMetadata(name="safe_delete_file", destructive=True),
            )
        )

    def test_tool_approval_respects_global_disable(self):
        self.assertFalse(
            tool_requires_approval(
                "edit_file",
                metadata=ToolMetadata(name="edit_file", mutating=True),
                approvals_enabled=False,
            )
        )

    def test_cli_exec_approval_uses_command_profile(self):
        self.assertFalse(tool_requires_approval("cli_exec", {"command": "Get-Process"}))
        self.assertFalse(tool_requires_approval("cli_exec", {"command": "rg --files"}))
        self.assertTrue(tool_requires_approval("cli_exec", {"command": "Remove-Item demo.txt"}))
        self.assertTrue(tool_requires_approval("cli_exec", {"command": ""}))

    def test_shell_approval_covers_mutating_aliases_in_every_pipeline_segment(self):
        """A read-only first segment must not vouch for a mutating alias later."""
        for command in (
            "Get-Content src.txt | ac target.txt",
            "Get-Content src.txt | ni target.txt",
            "Get-Content src.txt | tee target.txt",
            "Get-Content src.txt | sc target.txt",
            "Get-Content src.txt | cp a.txt b.txt",
            "Get-Content src.txt | clc target.txt",
            "Get-Content src.txt ; rm target.txt",
            "Get-Content src.txt & ni target.txt",
            "ls | ren a b",
            "cat src.txt | mv a b",
            "rg --files | tee files.txt",
        ):
            with self.subTest(command=command):
                self.assertTrue(shell_command_requires_approval(command), command)

    def test_shell_approval_covers_destructive_aliases_in_every_pipeline_segment(self):
        for command in (
            "Get-Content src.txt | ri src.txt",
            "Get-Content src.txt | del src.txt",
            "Get-Content src.txt | rd src.txt",
            "Get-Content src.txt | erase src.txt",
        ):
            with self.subTest(command=command):
                profile = classify_shell_command(command)
                self.assertTrue(shell_command_requires_approval(command), command)
                self.assertTrue(profile["destructive"], command)

    def test_shell_approval_covers_script_blocks_that_mutate(self):
        for command in (
            "Get-Content src.txt | % { iex $_ }",
            "Get-Content src.txt | %{ Add-Content out.txt 1 }",
            "ls | %{ ni demo.txt }",
            "$x = Get-Content src.txt | Remove-Item src.txt",
        ):
            with self.subTest(command=command):
                self.assertTrue(shell_command_requires_approval(command), command)

    def test_shell_approval_allows_read_only_pipelines_with_expressions(self):
        for command in (
            "Get-Process | Select-Object Id,ProcessName",
            "Get-ChildItem | Where-Object { $_.Length -gt 0 }",
            "rg --files 2>$null",
            "rg --files 2>&1",
            "Get-Content a.txt | Select-Object -First 3",
            "$x = 1",
            "$x = 'abc'",
            "$x = 1; Get-Process",
            "(Get-Process).Count",
            "ping google.com; curl -I https://ya.ru",
            "rg -n foo | Select-Object -First 5",
        ):
            with self.subTest(command=command):
                self.assertFalse(shell_command_requires_approval(command), command)

    def test_shell_approval_stays_conservative_for_literal_only_and_empty_commands(self):
        for command in ("", "   "):
            with self.subTest(command=command):
                profile = classify_shell_command(command)
                self.assertFalse(profile["inspect_only"], command)
                self.assertTrue(shell_command_requires_approval(command), command)

    def test_shell_approval_aliases_resolve_to_the_underlying_cmdlet(self):
        # ``ls``/``cat`` are aliases for the whitelisted Get-* cmdlets and stay
        # read-only; ``rm``/``ri`` resolve to Remove-Item and must not.
        for command in ("ls", "cat README.md", "gci", "gc README.md"):
            with self.subTest(command=command):
                self.assertFalse(shell_command_requires_approval(command), command)
        for command in ("rm demo.txt", "ri demo.txt", "del demo.txt"):
            with self.subTest(command=command):
                self.assertTrue(shell_command_requires_approval(command), command)


if __name__ == "__main__":
    unittest.main()
