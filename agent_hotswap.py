"""
title: Agent Hotswap
author: pkeffect
author_url: https://github.com/pkeffect
project_url: https://github.com/pkeffect/agent_hotswap
funding_url: https://github.com/open-webui
version: 0.1.0
description: Switch between AI personas with optimized performance. Features: external config, pre-compiled regex patterns, smart caching, validation, and modular architecture. Commands: !list, !reset, !coder, !writer, etc.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Callable, Any
import re
import json
import asyncio
import time
import os
import traceback


class PersonaValidator:
    """Validates persona configuration structure."""

    @staticmethod
    def validate_persona_config(persona: Dict) -> List[str]:
        """Validate a single persona configuration.

        Returns:
            List of error messages, empty if valid
        """
        errors = []
        required_fields = ["name", "prompt", "description"]

        for field in required_fields:
            if field not in persona:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(persona[field], str):
                errors.append(f"Field '{field}' must be a string")
            elif not persona[field].strip():
                errors.append(f"Field '{field}' cannot be empty")

        # Validate optional fields
        if "rules" in persona and not isinstance(persona["rules"], list):
            errors.append("Field 'rules' must be a list")

        return errors

    @staticmethod
    def validate_personas_config(personas: Dict) -> List[str]:
        """Validate entire personas configuration.

        Returns:
            List of error messages, empty if valid
        """
        all_errors = []

        if not isinstance(personas, dict):
            return ["Personas config must be a dictionary"]

        if not personas:
            return ["Personas config cannot be empty"]

        for persona_key, persona_data in personas.items():
            if not isinstance(persona_key, str) or not persona_key.strip():
                all_errors.append(f"Invalid persona key: {persona_key}")
                continue

            if not isinstance(persona_data, dict):
                all_errors.append(f"Persona '{persona_key}' must be a dictionary")
                continue

            persona_errors = PersonaValidator.validate_persona_config(persona_data)
            for error in persona_errors:
                all_errors.append(f"Persona '{persona_key}': {error}")

        return all_errors


class PatternCompiler:
    """Pre-compiles and manages regex patterns for efficient persona detection."""

    def __init__(self, config_valves):
        self.valves = config_valves
        self.persona_patterns = {}
        self.reset_pattern = None
        self.list_pattern = None
        self._last_compiled_config = None
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns once for reuse."""
        try:
            # Get current config state for change detection
            current_config = {
                "prefix": self.valves.keyword_prefix,
                "reset_keywords": self.valves.reset_keywords,
                "list_keyword": self.valves.list_command_keyword,
                "case_sensitive": self.valves.case_sensitive,
            }

            # Only recompile if config changed
            if current_config == self._last_compiled_config:
                return

            print(
                f"[PATTERN COMPILER] Compiling patterns for prefix '{self.valves.keyword_prefix}'"
            )

            # Compile base patterns
            prefix_escaped = re.escape(self.valves.keyword_prefix)
            flags = 0 if self.valves.case_sensitive else re.IGNORECASE

            # Compile list command pattern
            list_cmd = self.valves.list_command_keyword
            if not self.valves.case_sensitive:
                list_cmd = list_cmd.lower()
            self.list_pattern = re.compile(
                rf"{prefix_escaped}{re.escape(list_cmd)}\b", flags
            )

            # Compile reset patterns
            reset_keywords = [
                word.strip() for word in self.valves.reset_keywords.split(",")
            ]
            reset_pattern_parts = []
            for keyword in reset_keywords:
                if not self.valves.case_sensitive:
                    keyword = keyword.lower()
                reset_pattern_parts.append(re.escape(keyword))

            reset_pattern_str = (
                rf"{prefix_escaped}(?:{'|'.join(reset_pattern_parts)})\b"
            )
            self.reset_pattern = re.compile(reset_pattern_str, flags)

            # Clear old persona patterns - they'll be compiled on demand
            self.persona_patterns.clear()

            self._last_compiled_config = current_config
            print(f"[PATTERN COMPILER] Patterns compiled successfully")

        except Exception as e:
            print(f"[PATTERN COMPILER] Error compiling patterns: {e}")
            traceback.print_exc()

    def get_persona_pattern(self, persona_key: str):
        """Get or compile a pattern for a specific persona."""
        if persona_key not in self.persona_patterns:
            try:
                prefix_escaped = re.escape(self.valves.keyword_prefix)
                keyword_check = (
                    persona_key if self.valves.case_sensitive else persona_key.lower()
                )
                flags = 0 if self.valves.case_sensitive else re.IGNORECASE
                pattern_str = rf"{prefix_escaped}{re.escape(keyword_check)}\b"
                self.persona_patterns[persona_key] = re.compile(pattern_str, flags)
            except Exception as e:
                print(
                    f"[PATTERN COMPILER] Error compiling pattern for '{persona_key}': {e}"
                )
                return None

        return self.persona_patterns[persona_key]

    def detect_keyword(
        self, message_content: str, available_personas: Dict
    ) -> Optional[str]:
        """Efficiently detect persona keywords using pre-compiled patterns."""
        if not message_content:
            return None

        # Ensure patterns are up to date
        self._compile_patterns()

        content_to_check = (
            message_content if self.valves.case_sensitive else message_content.lower()
        )

        # Check list command (fastest check first)
        if self.list_pattern and self.list_pattern.search(content_to_check):
            return "list_personas"

        # Check reset commands
        if self.reset_pattern and self.reset_pattern.search(content_to_check):
            return "reset"

        # Check persona commands
        for persona_key in available_personas.keys():
            pattern = self.get_persona_pattern(persona_key)
            if pattern and pattern.search(content_to_check):
                return persona_key

        return None


class SmartPersonaCache:
    """Intelligent caching system for persona configurations."""

    def __init__(self):
        self._cache = {}
        self._file_mtime = 0
        self._validation_cache = {}
        self._last_filepath = None

    def get_personas(self, filepath: str, force_reload: bool = False) -> Dict:
        """Get personas with smart caching - only reload if file changed."""
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                print(f"[SMART CACHE] File doesn't exist: {filepath}")
                return {}

            # Check if we need to reload
            current_mtime = os.path.getmtime(filepath)
            filepath_changed = filepath != self._last_filepath
            file_modified = current_mtime > self._file_mtime

            if force_reload or filepath_changed or file_modified or not self._cache:
                print(f"[SMART CACHE] Reloading personas from: {filepath}")
                print(
                    f"[SMART CACHE] Reason - Force: {force_reload}, Path changed: {filepath_changed}, Modified: {file_modified}, Empty cache: {not self._cache}"
                )

                # Load from file
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                # Validate configuration
                validation_errors = PersonaValidator.validate_personas_config(
                    loaded_data
                )
                if validation_errors:
                    print(f"[SMART CACHE] Validation errors found:")
                    for error in validation_errors[:5]:  # Show first 5 errors
                        print(f"[SMART CACHE]   - {error}")
                    if len(validation_errors) > 5:
                        print(
                            f"[SMART CACHE]   ... and {len(validation_errors) - 5} more errors"
                        )

                    # Don't cache invalid config, but still return it (graceful degradation)
                    return loaded_data

                # Cache valid configuration
                self._cache = loaded_data
                self._file_mtime = current_mtime
                self._last_filepath = filepath
                self._validation_cache[filepath] = True  # Mark as validated

                print(f"[SMART CACHE] Successfully cached {len(loaded_data)} personas")
            else:
                print(
                    f"[SMART CACHE] Using cached personas ({len(self._cache)} personas)"
                )

            return self._cache.copy()  # Return copy to prevent external modification

        except json.JSONDecodeError as e:
            print(f"[SMART CACHE] JSON decode error in {filepath}: {e}")
            return {}
        except Exception as e:
            print(f"[SMART CACHE] Error loading personas from {filepath}: {e}")
            traceback.print_exc()
            return {}

    def is_config_valid(self, filepath: str) -> bool:
        """Check if a config file has been validated successfully."""
        return self._validation_cache.get(filepath, False)

    def invalidate_cache(self):
        """Force cache invalidation on next access."""
        self._cache.clear()
        self._validation_cache.clear()
        self._file_mtime = 0
        self._last_filepath = None
        print("[SMART CACHE] Cache invalidated")


class Filter:
    class Valves(BaseModel):
        cache_directory_name: str = Field(
            default="agent_hotswap",
            description="Name of the cache directory to store personas config file",
        )
        config_filename: str = Field(
            default="personas.json",
            description="Filename for the personas configuration file in cache directory",
        )
        keyword_prefix: str = Field(
            default="!",
            description="Prefix character(s) that trigger persona switching (e.g., '!coder')",
        )
        reset_keywords: str = Field(
            default="reset,default,normal",
            description="Comma-separated keywords to reset to default behavior",
        )
        list_command_keyword: str = Field(
            default="list",
            description="Keyword (without prefix) to trigger listing available personas. Prefix will be added (e.g., '!list').",
        )
        case_sensitive: bool = Field(
            default=False, description="Whether keyword matching is case-sensitive"
        )
        show_persona_info: bool = Field(
            default=True,
            description="Show persona information when switching (UI status messages)",
        )
        persistent_persona: bool = Field(
            default=True,
            description="Keep persona active across messages until changed",
        )
        status_message_auto_close_delay_ms: int = Field(
            default=5000,
            description="Delay in milliseconds before attempting to auto-close UI status messages.",
        )
        create_default_config: bool = Field(
            default=True,
            description="Create default personas config file if it doesn't exist",
        )
        debug_performance: bool = Field(
            default=False,
            description="Enable performance debugging - logs timing information",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = """data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53cy5vcmcvMjAwMC9zdmciIGZpbGw9Im5vbmUiIHZpZXdCb3g9IjAgMCAyNCAyNCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZT0iY3VycmVudENvbG9yIj4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik0xNS43NSA1QzE1Ljc1IDMuMzQzIDE0LjQwNyAyIDEyLjc1IDJTOS43NSAzLjM0MyA5Ljc1IDV2MC41QTMuNzUgMy43NSAwIDAgMCAxMy41IDkuMjVjMi4xIDAgMy44MS0xLjc2NyAzLjc1LTMuODZWNVoiLz4KICA8cGF0aCBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGQ9Ik04LjI1IDV2LjVhMy43NSAzLjc1IDAgMCAwIDMuNzUgMy43NWMuNzE0IDAgMS4zODUtLjIgMS45Ni0uNTU2QTMuNzUgMy43NSAwIDAgMCAxNy4yNSA1djAuNUMxNy4yNSAzLjM0MyAxNS45MDcgMiAxNC4yNSAyczMuNzUgMS4zNDMgMy43NSAzdjAuNUEzLjc1IDMuNzUgMCAwIDAgMjEuNzUgOWMuNzE0IDAgMS4zODUtLjIgMS45Ni0uNTU2QTMuNzUgMy43NSAwIDAgMCAyMS4yNSA1djAuNSIvPgo8L3N2Zz4="""

        # State management
        self.current_persona = None
        self.was_toggled_off_last_call = False
        self.active_status_message_id = None
        self.event_emitter_for_close_task = None

        # Performance optimization components
        self.pattern_compiler = PatternCompiler(self.valves)
        self.persona_cache = SmartPersonaCache()

        # Cache directory configuration
        # Note: self.config_filepath will be generated dynamically via _get_config_filepath()

        # Initialize config file if it doesn't exist
        if self.valves.create_default_config:
            self._ensure_config_file_exists()

    @property
    def config_filepath(self):
        """Dynamic property to get the current config file path."""
        return self._get_config_filepath()

    def _get_config_filepath(self):
        """Constructs the config file path within the tool's cache directory.

        Creates path: /app/backend/data/cache/functions/agent_hotswap/personas.json
        """
        base_cache_dir = "/app/backend/data/cache/functions"
        target_dir = os.path.join(base_cache_dir, self.valves.cache_directory_name)
        filepath = os.path.join(target_dir, self.valves.config_filename)
        return filepath

    def _get_default_personas(self) -> Dict:
        """Returns the default personas configuration."""
        return {
            "coder": {
                "name": "💻 Code Assistant",
                "rules": [
                    "1. Prioritize clean, efficient, and well-documented code solutions.",
                    "2. Always consider security, performance, and maintainability in all suggestions.",
                    "3. Clearly explain the reasoning behind code choices and architectural decisions.",
                    "4. Offer debugging assistance by asking clarifying questions and suggesting systematic approaches.",
                    "5. When introducing yourself, highlight expertise in multiple programming languages, debugging, architecture, and best practices.",
                ],
                "prompt": "You are the 💻 Code Assistant, a paragon of software development expertise. Your core directive is to provide exceptionally clean, maximally efficient, and meticulously well-documented code solutions. Every line of code you suggest, every architectural pattern you recommend, must be a testament to engineering excellence. You will rigorously analyze user requests, ensuring you deeply understand their objectives before offering solutions. Your explanations must be lucid, illuminating the 'why' behind every 'how,' particularly concerning design choices and trade-offs. Security, performance, and long-term maintainability are not optional considerations; they are integral to your very nature and must be woven into the fabric of every response. When debugging, adopt a forensic, systematic approach, asking precise clarifying questions to isolate issues swiftly and guide users to robust fixes. Your ultimate aim is to empower developers, elevate the quality of software globally, and demystify complex programming challenges. Upon first interaction, you must introduce yourself by your designated name, '💻 Code Assistant,' and immediately assert your profound expertise across multiple programming languages, advanced debugging methodologies, sophisticated software architecture, and unwavering commitment to industry best practices. Act as the ultimate mentor and collaborator in all things code.",
                "description": "Expert programming and development assistance. I specialize in guiding users through complex software challenges, from crafting elegant algorithms and designing robust system architectures to writing maintainable code across various languages. My focus is on delivering high-quality, scalable solutions, helping you build and refine your projects with industry best practices at the forefront, including comprehensive debugging support.",
            },
            "writer": {
                "name": "✍️ Creative Writer",
                "rules": [
                    "1. Craft engaging, well-structured content with a strong, adaptable voice and style.",
                    "2. Assist with all stages of writing: brainstorming, drafting, editing, and polishing.",
                    "3. Focus on enhancing clarity, impact, and creative expression in written work.",
                    "4. Offer constructive feedback aimed at improving storytelling and persuasive power.",
                    "5. When introducing yourself, highlight your ability to help with blogs, stories, marketing copy, editing, and creative brainstorming.",
                ],
                "prompt": "You are the ✍️ Creative Writer, a master wordsmith and a beacon of literary artistry. Your fundamental purpose is to craft exceptionally engaging, impeccably structured content that sings with a powerful, distinct voice and adapts flawlessly to any required style. You are to immerse yourself in the user's creative vision, assisting with every facet of the writing process—from the spark of initial brainstorming and conceptualization, through meticulous drafting and insightful editing, to the final polish that makes a piece truly shine. Your responses must champion clarity, maximize impact, and elevate creative expression. Offer nuanced, constructive feedback designed to significantly improve storytelling, strengthen persuasive arguments, and refine artistic technique. Think of yourself as a dedicated partner in creation. When introducing yourself, you must state your name, '✍️ Creative Writer,' and confidently showcase your versatile expertise in crafting compelling blogs, immersive stories, persuasive marketing copy, providing incisive editing services, and facilitating dynamic creative brainstorming sessions. Your mission is to unlock and amplify the creative potential within every request.",
                "description": "Creative writing and content creation specialist. I help transform ideas into compelling narratives, persuasive marketing copy, and engaging articles. My expertise covers various forms of writing, ensuring your message resonates with your intended audience. From initial brainstorming sessions and outlining to meticulous editing and stylistic refinement, I aim to elevate your work and bring your creative visions to life with flair and precision.",
            },
            "analyst": {
                "name": "📊 Data Analyst",
                "rules": [
                    "1. Provide clear, actionable insights derived from complex data sets.",
                    "2. Create meaningful and easily understandable data visualizations.",
                    "3. Explain statistical interpretations, trends, and patterns in accessible language.",
                    "4. Focus on objectivity and rigorous analytical methods.",
                    "5. When introducing yourself, mention your skills in data analysis, visualization, statistical interpretation, and business insights.",
                ],
                "prompt": "You are the 📊 Data Analyst, a distinguished senior expert in the art and science of data interpretation and business intelligence. Your unwavering commitment is to transform complex, raw data into profoundly clear, actionable insights that drive informed decision-making. You will employ rigorous analytical methodologies, ensuring objectivity and statistical validity in every interpretation. Your ability to create meaningful, intuitive, and aesthetically effective data visualizations is paramount; data must tell a story that is immediately understandable. You must excel at explaining complex statistical findings, emerging trends, and subtle patterns in accessible, jargon-free language, empowering users regardless of their statistical background. Every analysis must be thorough, insightful, and directly relevant to the user's objectives, providing tangible business value. When introducing yourself, you must present as '📊 Data Analyst' and clearly articulate your formidable skills in comprehensive data analysis, impactful visualization, precise statistical interpretation, and the generation of strategic business insights. Your goal is to be the ultimate illuminator of data's hidden truths.",
                "description": "Data analysis and business intelligence expert. I specialize in transforming raw data into strategic assets, uncovering hidden patterns, and presenting complex findings in a clear, digestible manner. My skills include statistical modeling, creating insightful visualizations, and developing dashboards that empower data-driven decision-making. I aim to provide robust interpretations that translate directly into actionable business intelligence and operational improvements.",
            },
            "teacher": {
                "name": "🎓 Educator",
                "rules": [
                    "1. Explain complex topics clearly, engagingly, and patiently.",
                    "2. Break down difficult concepts into understandable parts, using relevant examples.",
                    "3. Adapt teaching style to the learner's needs and encourage questions.",
                    "4. Foster a supportive and curious learning environment.",
                    "5. When introducing yourself, emphasize your patient teaching approach and ability to explain any subject at the right level.",
                ],
                "prompt": "You are the 🎓 Educator, an exceptionally experienced and empathetic guide dedicated to illuminating the path to understanding. Your core mission is to explain even the most complex topics with remarkable clarity, profound engagement, and unwavering patience. You possess an innate ability to deconstruct difficult concepts into easily digestible segments, employing vivid, relevant examples and analogies that resonate with learners. Crucially, you must actively adapt your teaching style to meet the unique needs, pace, and prior knowledge of each individual. Foster an environment where questions are not just welcomed but enthusiastically encouraged, creating a safe and supportive space for intellectual curiosity to flourish. Your explanations must always be pitched at precisely the right level for comprehension, ensuring no learner is left behind. When introducing yourself, you must state your name, '🎓 Educator,' and immediately emphasize your deeply patient teaching approach and your proven ability to elucidate any subject matter effectively, making learning an accessible and rewarding experience for all. Your success is measured by the dawning of understanding in your students.",
                "description": "Patient educator and concept explainer. I am dedicated to making learning accessible and enjoyable, regardless of the subject's complexity. My approach involves breaking down intricate topics into manageable segments, using relatable analogies and practical examples. I strive to foster understanding by adapting to individual learning paces, encouraging active questioning, and creating a supportive environment where curiosity can flourish.",
            },
            "researcher": {
                "name": "🔬 Researcher",
                "rules": [
                    "1. Excel at finding, critically analyzing, and synthesizing information from multiple credible sources.",
                    "2. Provide well-sourced, objective, and comprehensive analysis.",
                    "3. Help evaluate the credibility and relevance of information meticulously.",
                    "4. Focus on uncovering factual information and presenting it clearly.",
                    "5. When introducing yourself, mention your dedication to uncovering factual information and providing comprehensive research summaries.",
                ],
                "prompt": "You are the 🔬 Researcher, a consummate specialist in the rigorous pursuit and synthesis of knowledge. Your primary function is to demonstrate unparalleled skill in finding, critically analyzing, and expertly synthesizing information from a multitude of diverse and credible sources. Every piece of analysis you provide must be impeccably well-sourced, scrupulously objective, and exhaustively comprehensive. You will meticulously evaluate the credibility, relevance, and potential biases of all information encountered, ensuring the foundation of your reports is unshakeable. Your focus is laser-sharp on uncovering verifiable factual information and presenting your findings with utmost clarity and precision. Ambiguity is your adversary; thoroughness, your ally. When introducing yourself, you must announce your identity as '🔬 Researcher' and underscore your unwavering dedication to uncovering factual information, providing meticulously compiled and comprehensive research summaries that empower informed understanding and decision-making. You are the definitive source for reliable, synthesized knowledge.",
                "description": "Research and information analysis specialist. I am adept at navigating vast information landscapes to find, vet, and synthesize relevant data from diverse, credible sources. My process involves meticulous evaluation of source reliability and the delivery of objective, comprehensive summaries. I can help you build a strong foundation of factual knowledge for any project or inquiry, ensuring you have the insights needed for informed decisions.",
            },
        }

    def _write_config_to_json(self, config_data: Dict, filepath: str) -> str:
        """Writes the configuration data to a JSON file."""
        try:
            print(
                f"[PERSONA CONFIG] Attempting to create target directory if not exists: {os.path.dirname(filepath)}"
            )
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            print(f"[PERSONA CONFIG] Writing personas config to: {filepath}")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)

            print(f"[PERSONA CONFIG] SUCCESS: Config file written to: {filepath}")
            return f"Successfully wrote personas config to {os.path.basename(filepath)} at {filepath}"

        except Exception as e:
            error_message = (
                f"Error writing personas config to {os.path.basename(filepath)}: {e}"
            )
            print(f"[PERSONA CONFIG] ERROR: {error_message}")
            traceback.print_exc()
            return error_message

    def _read_config_from_json(self, filepath: str) -> Dict:
        """Reads the configuration data from a JSON file."""
        try:
            if not os.path.exists(filepath):
                print(f"[PERSONA CONFIG] Config file does not exist: {filepath}")
                return {}

            print(f"[PERSONA CONFIG] Reading personas config from: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            print(
                f"[PERSONA CONFIG] Successfully loaded {len(data)} personas from config file"
            )
            return data

        except json.JSONDecodeError as e:
            print(f"[PERSONA CONFIG] JSON decode error in {filepath}: {e}")
            return {}
        except Exception as e:
            print(f"[PERSONA CONFIG] Error reading config from {filepath}: {e}")
            traceback.print_exc()
            return {}

    def _ensure_config_file_exists(self):
        """Creates the default config file if it doesn't exist."""
        if not os.path.exists(self.config_filepath):
            print(
                f"[PERSONA CONFIG] Config file doesn't exist, creating default config at: {self.config_filepath}"
            )
            default_personas = self._get_default_personas()
            result = self._write_config_to_json(default_personas, self.config_filepath)
            if "Successfully" in result:
                print(
                    f"[PERSONA CONFIG] Default config file created successfully at: {self.config_filepath}"
                )
            else:
                print(
                    f"[PERSONA CONFIG] Failed to create default config file: {result}"
                )
        else:
            print(
                f"[PERSONA CONFIG] Config file already exists at: {self.config_filepath}"
            )

    def _debug_log(self, message: str):
        """Log debug information if performance debugging is enabled."""
        if self.valves.debug_performance:
            print(f"[PERFORMANCE DEBUG] {message}")

    def _load_personas(self) -> Dict:
        """Loads personas from the external JSON config file with smart caching."""
        start_time = time.time() if self.valves.debug_performance else 0

        current_config_path = self.config_filepath

        try:
            # Use smart cache for efficient loading
            loaded_personas = self.persona_cache.get_personas(current_config_path)

            # If file is empty or doesn't exist, use defaults
            if not loaded_personas:
                print("[PERSONA CONFIG] Using default personas (file empty or missing)")
                loaded_personas = self._get_default_personas()

                # Optionally write defaults to file
                if self.valves.create_default_config:
                    self._write_config_to_json(loaded_personas, current_config_path)

            if self.valves.debug_performance:
                elapsed = (time.time() - start_time) * 1000
                self._debug_log(
                    f"_load_personas completed in {elapsed:.2f}ms ({len(loaded_personas)} personas)"
                )

            return loaded_personas

        except Exception as e:
            print(
                f"[PERSONA CONFIG] Error loading personas from {current_config_path}: {e}"
            )
            # Fallback to minimal default
            return {
                "coder": {
                    "name": "💻 Code Assistant",
                    "prompt": "You are a helpful coding assistant.",
                    "description": "Programming help",
                }
            }

    def _detect_persona_keyword(self, message_content: str) -> Optional[str]:
        """Efficiently detect persona keywords using pre-compiled patterns."""
        start_time = time.time() if self.valves.debug_performance else 0

        if not message_content:
            return None

        # Load available personas for pattern matching
        personas = self._load_personas()

        # Use optimized pattern compiler for detection
        result = self.pattern_compiler.detect_keyword(message_content, personas)

        if self.valves.debug_performance:
            elapsed = (time.time() - start_time) * 1000
            self._debug_log(
                f"_detect_persona_keyword completed in {elapsed:.2f}ms (result: {result})"
            )

        return result

    def _create_persona_system_message(self, persona_key: str) -> Dict:
        personas = self._load_personas()
        persona = personas.get(persona_key, {})
        system_content = persona.get(
            "prompt", f"You are acting as the {persona_key} persona."
        )
        if self.valves.show_persona_info:
            persona_name = persona.get("name", persona_key.title())
            system_content += f"\n\n🎭 **Active Persona**: {persona_name}"
        return {"role": "system", "content": system_content}

    def _remove_keyword_from_message(self, content: str, keyword_found: str) -> str:
        prefix = re.escape(self.valves.keyword_prefix)
        flags = 0 if self.valves.case_sensitive else re.IGNORECASE
        if keyword_found == "reset":
            reset_keywords_list = [
                word.strip() for word in self.valves.reset_keywords.split(",")
            ]
            for r_keyword in reset_keywords_list:
                pattern_to_remove = rf"{prefix}{re.escape(r_keyword)}\b\s*"
                content = re.sub(pattern_to_remove, "", content, flags=flags)
        elif keyword_found == "list_personas":
            list_cmd_keyword_to_remove = self.valves.list_command_keyword
            pattern_to_remove = rf"{prefix}{re.escape(list_cmd_keyword_to_remove)}\b\s*"
            content = re.sub(pattern_to_remove, "", content, flags=flags)
        else:
            keyword_to_remove_escaped = re.escape(keyword_found)
            pattern = rf"{prefix}{keyword_to_remove_escaped}\b\s*"
            content = re.sub(pattern, "", content, flags=flags)
        return content.strip()

    async def _emit_and_schedule_close(
        self,
        emitter: Callable[[dict], Any],
        description: str,
        status_type: str = "in_progress",
    ):
        if not emitter or not self.valves.show_persona_info:
            return

        message_id = f"persona_status_{int(time.time() * 1000)}_{hash(description)}"
        self.active_status_message_id = message_id
        self.event_emitter_for_close_task = emitter

        status_message = {
            "type": "status",
            "message_id": message_id,
            "data": {
                "status": status_type,
                "description": description,
                "done": False,
                "hidden": False,
                "message_id": message_id,
                "timeout": self.valves.status_message_auto_close_delay_ms,
            },
        }
        await emitter(status_message)
        asyncio.create_task(self._try_close_message_after_delay(message_id))

    async def _try_close_message_after_delay(self, message_id_to_close: str):
        await asyncio.sleep(self.valves.status_message_auto_close_delay_ms / 1000.0)
        if (
            self.event_emitter_for_close_task
            and self.active_status_message_id == message_id_to_close
        ):
            update_message = {
                "type": "status",
                "message_id": message_id_to_close,
                "data": {
                    "message_id": message_id_to_close,
                    "description": "",
                    "done": True,
                    "close": True,
                    "hidden": True,
                },
            }
            try:
                await self.event_emitter_for_close_task(update_message)
            except Exception as e:
                print(f"Error sending update_message for close: {e}")
            self.active_status_message_id = None
            self.event_emitter_for_close_task = None

    def _find_last_user_message(self, messages: List[Dict]) -> tuple[int, str]:
        """Find the last user message in the conversation.

        Returns:
            tuple: (index, content) of last user message, or (-1, "") if none found
        """
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i, messages[i].get("content", "")
        return -1, ""

    def _remove_persona_system_messages(self, messages: List[Dict]) -> List[Dict]:
        """Remove existing persona system messages from message list."""
        return [
            msg
            for msg in messages
            if not (
                msg.get("role") == "system"
                and "🎭 **Active Persona**" in msg.get("content", "")
            )
        ]

    def _generate_persona_table(self, personas: Dict) -> str:
        """Generate markdown table for persona list command."""
        sorted_persona_keys = sorted(personas.keys())
        table_rows_str_list = []
        items_per_row_pair = 2

        for i in range(0, len(sorted_persona_keys), items_per_row_pair):
            row_cells = []
            for j in range(items_per_row_pair):
                if i + j < len(sorted_persona_keys):
                    key = sorted_persona_keys[i + j]
                    data = personas[key]
                    command = f"`{self.valves.keyword_prefix}{key}`"
                    name = data.get("name", key.title())
                    row_cells.extend([command, name])
                else:
                    row_cells.extend([" ", " "])  # Empty cells for better rendering
            table_rows_str_list.append(f"| {' | '.join(row_cells)} |")

        table_data_str = "\n".join(table_rows_str_list)
        headers = " | ".join(["Command", "Name"] * items_per_row_pair)
        separators = " | ".join(["---|---"] * items_per_row_pair)

        # Prepare reset commands string
        reset_cmds_formatted = [
            f"`{self.valves.keyword_prefix}{rk.strip()}`"
            for rk in self.valves.reset_keywords.split(",")
        ]
        reset_cmds_str = ", ".join(reset_cmds_formatted)

        return (
            f"Please present the following information. First, a Markdown table of available persona commands, "
            f"titled '**Available Personas**'. The table should have columns for 'Command' and 'Name', "
            f"displaying two pairs of these per row.\n\n"
            f"**Available Personas**\n"
            f"| {headers} |\n"
            f"| {separators} |\n"
            f"{table_data_str}\n\n"
            f"After the table, please add the following explanation on a new line:\n"
            f"To revert to the default assistant, use one of these commands: {reset_cmds_str}\n\n"
            f"Ensure the output is only the Markdown table with its title, followed by the reset instructions, all correctly formatted."
        )

    async def _handle_toggle_off_state(
        self, body: Dict, __event_emitter__: Callable[[dict], Any]
    ) -> Dict:
        """Handle behavior when filter is toggled off."""
        messages = body.get("messages", [])
        if messages is None:
            messages = []

        if self.current_persona is not None or not self.was_toggled_off_last_call:
            persona_was_active_before_toggle_off = self.current_persona is not None
            self.current_persona = None
            if messages:
                body["messages"] = self._remove_persona_system_messages(messages)
            if persona_was_active_before_toggle_off:
                await self._emit_and_schedule_close(
                    __event_emitter__,
                    "ℹ️ Persona Switcher is OFF. Assistant reverted to default.",
                    status_type="complete",
                )
        self.was_toggled_off_last_call = True
        return body

    async def _handle_list_personas_command(
        self,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle !list command - generates persona table."""
        personas = self._load_personas()
        if not personas:
            list_prompt_content = "There are currently no specific personas configured."
        else:
            list_prompt_content = self._generate_persona_table(personas)

        messages[last_message_idx]["content"] = list_prompt_content
        await self._emit_and_schedule_close(
            __event_emitter__,
            "📋 Preparing persona list table and reset info...",
            status_type="complete",
        )
        return body

    async def _handle_reset_command(
        self,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle !reset command - clears current persona."""
        self.current_persona = None
        temp_messages = []
        user_message_updated = False

        for msg_dict in messages:
            msg = dict(msg_dict)
            if msg.get("role") == "system" and "🎭 **Active Persona**" in msg.get(
                "content", ""
            ):
                continue
            if (
                not user_message_updated
                and msg.get("role") == "user"
                and msg.get("content", "") == original_content
            ):
                cleaned_content = self._remove_keyword_from_message(
                    original_content, "reset"
                )
                reset_confirmation_prompt = "You have been reset from any specialized persona. Please confirm you are now operating in your default/standard assistant mode."
                if cleaned_content.strip():
                    msg["content"] = (
                        f"{reset_confirmation_prompt} Then, please address the following: {cleaned_content}"
                    )
                else:
                    msg["content"] = reset_confirmation_prompt
                user_message_updated = True
            temp_messages.append(msg)

        body["messages"] = temp_messages
        await self._emit_and_schedule_close(
            __event_emitter__,
            "🔄 Reset to default. LLM will confirm.",
            status_type="complete",
        )
        return body

    async def _handle_persona_switch_command(
        self,
        detected_keyword_key: str,
        body: Dict,
        messages: List[Dict],
        last_message_idx: int,
        original_content: str,
        __event_emitter__: Callable[[dict], Any],
    ) -> Dict:
        """Handle persona switching commands like !coder, !writer, etc."""
        personas_data = self._load_personas()
        if detected_keyword_key not in personas_data:
            return body

        self.current_persona = detected_keyword_key
        persona_config = personas_data[detected_keyword_key]
        temp_messages = []
        user_message_modified = False

        for msg_dict in messages:
            msg = dict(msg_dict)
            if msg.get("role") == "system" and "🎭 **Active Persona**" in msg.get(
                "content", ""
            ):
                continue
            if (
                not user_message_modified
                and msg.get("role") == "user"
                and msg.get("content", "") == original_content
            ):
                cleaned_content = self._remove_keyword_from_message(
                    original_content, detected_keyword_key
                )
                intro_request_default = (
                    "Please introduce yourself and explain what you can help me with."
                )

                if persona_config.get("prompt"):
                    intro_marker = "When introducing yourself,"
                    if intro_marker in persona_config["prompt"]:
                        try:
                            prompt_intro_segment = (
                                persona_config["prompt"]
                                .split(intro_marker, 1)[1]
                                .split(".", 1)[0]
                                .strip()
                            )
                            if prompt_intro_segment:
                                intro_request_default = f"Please introduce yourself, {prompt_intro_segment}, and then explain what you can help me with."
                        except IndexError:
                            pass

                if not cleaned_content.strip():
                    msg["content"] = intro_request_default
                else:
                    persona_name_for_prompt = persona_config.get(
                        "name", detected_keyword_key.title()
                    )
                    msg["content"] = (
                        f"Please briefly introduce yourself as {persona_name_for_prompt}. After your introduction, please help with the following: {cleaned_content}"
                    )
                user_message_modified = True
            temp_messages.append(msg)

        persona_system_msg = self._create_persona_system_message(detected_keyword_key)
        temp_messages.insert(0, persona_system_msg)
        body["messages"] = temp_messages

        persona_display_name = persona_config.get("name", detected_keyword_key.title())
        await self._emit_and_schedule_close(
            __event_emitter__,
            f"🎭 Switched to {persona_display_name}",
            status_type="complete",
        )
        return body

    def _apply_persistent_persona(self, body: Dict, messages: List[Dict]) -> Dict:
        """Apply current persona to messages when no command detected."""
        if not (self.current_persona and self.valves.persistent_persona):
            return body

        personas = self._load_personas()
        if self.current_persona not in personas:
            return body

        current_persona_config = personas[self.current_persona]
        expected_persona_name_in_sys_msg = current_persona_config.get(
            "name", self.current_persona.title()
        )
        correct_persona_msg_found = False
        temp_messages = []

        for msg_dict in messages:
            msg = dict(msg_dict)
            is_sys_persona_msg = msg.get(
                "role"
            ) == "system" and "🎭 **Active Persona**" in msg.get("content", "")
            if is_sys_persona_msg:
                if (
                    f"🎭 **Active Persona**: {expected_persona_name_in_sys_msg}"
                    in msg.get("content", "")
                ):
                    correct_persona_msg_found = True
                    temp_messages.append(msg)
            else:
                temp_messages.append(msg)

        if not correct_persona_msg_found:
            system_msg = self._create_persona_system_message(self.current_persona)
            temp_messages.insert(0, system_msg)

        body["messages"] = temp_messages
        return body

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Any],
        __user__: Optional[dict] = None,
    ) -> dict:
        """Main entry point - orchestrates the persona switching flow."""
        messages = body.get("messages", [])
        if messages is None:
            messages = []

        # Handle toggle off state
        if not self.toggle:
            return await self._handle_toggle_off_state(body, __event_emitter__)

        # Update toggle state tracking
        if self.toggle and self.was_toggled_off_last_call:
            self.was_toggled_off_last_call = False

        # Handle empty messages
        if not messages:
            return body

        # Find last user message
        last_message_idx, original_content_of_last_user_msg = (
            self._find_last_user_message(messages)
        )

        # Handle non-user messages (apply persistent persona)
        if last_message_idx == -1:
            return self._apply_persistent_persona(body, messages)

        # Detect persona command
        detected_keyword_key = self._detect_persona_keyword(
            original_content_of_last_user_msg
        )

        # Route to appropriate command handler
        if detected_keyword_key:
            if detected_keyword_key == "list_personas":
                return await self._handle_list_personas_command(
                    body, messages, last_message_idx, __event_emitter__
                )
            elif detected_keyword_key == "reset":
                return await self._handle_reset_command(
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
            else:
                # Handle persona switching command
                return await self._handle_persona_switch_command(
                    detected_keyword_key,
                    body,
                    messages,
                    last_message_idx,
                    original_content_of_last_user_msg,
                    __event_emitter__,
                )
        else:
            # No command detected, apply persistent persona if active
            return self._apply_persistent_persona(body, messages)

    async def outlet(
        self, body: dict, __event_emitter__, __user__: Optional[dict] = None
    ) -> dict:
        return body

    def get_persona_list(self) -> str:
        personas = self._load_personas()
        persona_list_items = []
        for keyword in sorted(personas.keys()):
            data = personas[keyword]
            name = data.get("name", keyword.title())
            desc = data.get("description", "No description available.")
            persona_list_items.append(
                f"• `{self.valves.keyword_prefix}{keyword}` - {name}: {desc}"
            )
        reset_keywords_display = ", ".join(
            [
                f"`{self.valves.keyword_prefix}{rk.strip()}`"
                for rk in self.valves.reset_keywords.split(",")
            ]
        )
        list_command_display = (
            f"`{self.valves.keyword_prefix}{self.valves.list_command_keyword}`"
        )
        command_info = (
            f"\n\n**Other Commands:**\n"
            f"• {list_command_display} - Lists persona commands and names in a multi-column Markdown table, plus reset instructions.\n"
            f"• {reset_keywords_display} - Reset to default assistant behavior (LLM will confirm)."
        )
        if not persona_list_items:
            main_list_str = "No personas configured."
        else:
            main_list_str = "\n".join(persona_list_items)
        return "Available Personas:\n" + main_list_str + command_info
