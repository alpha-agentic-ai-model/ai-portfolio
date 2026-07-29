import asyncio
import base64
import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from PIL import Image, ImageDraw, ImageFont
import pyautogui
import time


class ActionType(Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY = "key"
    WAIT = "wait"
    DRAG = "drag"


@dataclass
class UIElement:
    label: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    element_type: str  # button, input, text, image, etc.
    confidence: float
    text_content: str = ""
    is_interactive: bool = True


@dataclass
class Action:
    action_type: ActionType
    target_element: Optional[int] = None  # SoM index
    text: str = ""
    coordinates: Optional[tuple[int, int]] = None
    expected_state: str = ""
    reasoning: str = ""


@dataclass
class StepResult:
    action: Action
    success: bool
    before_screenshot: Optional[Image.Image] = None
    after_screenshot: Optional[Image.Image] = None
    error: str = ""


class VisionGrounding:
    """UI element detection and Set-of-Mark annotation."""

    def __init__(self, model):
        self.model = model
        self.colors = [
            (255, 107, 107), (108, 99, 255), (0, 212, 255),
            (255, 217, 61), (74, 222, 128), (255, 154, 0),
        ]

    async def detect_ui(self, screenshot: Image.Image) -> list[UIElement]:
        """Detect UI elements using vision model."""
        img_bytes = io.BytesIO()
        screenshot.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode()

        response = await self.model.analyze(
            image=img_b64,
            prompt="Detect all interactive UI elements: buttons, inputs, links, menus, checkboxes, dropdowns. "
                   "Return bounding boxes and element types.",
        )
        return self._parse_detections(response)

    def annotate_set_of_mark(self, screenshot: Image.Image, elements: list[UIElement]) -> Image.Image:
        """Annotate screenshot with numbered markers on each UI element."""
        annotated = screenshot.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()

        for i, el in enumerate(elements):
            color = self.colors[i % len(self.colors)]
            x1, y1, x2, y2 = el.bbox

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw numbered marker
            marker_x, marker_y = x1, y1 - 18
            marker_w, marker_h = 22, 18
            draw.rectangle(
                [marker_x, marker_y, marker_x + marker_w, marker_y + marker_h],
                fill=color,
            )
            draw.text((marker_x + 4, marker_y + 1), str(i + 1), fill=(255, 255, 255), font=font)

        return annotated

    def _parse_detections(self, response: Any) -> list[UIElement]:
        """Parse model response into UIElement list."""
        elements = []
        for det in response.detections:
            elements.append(UIElement(
                label=det.label,
                bbox=tuple(det.bbox),
                element_type=det.category,
                confidence=det.score,
                text_content=det.text or "",
                is_interactive=det.category in {"button", "input", "link", "checkbox", "dropdown", "tab"},
            ))
        return elements


class ActionExecutor:
    """Execute UI actions via pyautogui."""

    def __init__(self, safety_pause: float = 0.3):
        pyautogui.PAUSE = safety_pause
        pyautogui.FAILSAFE = True

    async def capture(self) -> Image.Image:
        return pyautogui.screenshot()

    async def execute(self, action: Action, elements: list[UIElement]) -> StepResult:
        """Execute a planned action."""
        before = await self.capture()
        try:
            if action.target_element is not None and action.target_element <= len(elements):
                el = elements[action.target_element - 1]
                x = (el.bbox[0] + el.bbox[2]) // 2
                y = (el.bbox[1] + el.bbox[3]) // 2
            elif action.coordinates:
                x, y = action.coordinates
            else:
                x, y = None, None

            if action.action_type == ActionType.CLICK and x is not None:
                pyautogui.click(x, y)
            elif action.action_type == ActionType.DOUBLE_CLICK and x is not None:
                pyautogui.doubleClick(x, y)
            elif action.action_type == ActionType.RIGHT_CLICK and x is not None:
                pyautogui.rightClick(x, y)
            elif action.action_type == ActionType.TYPE:
                pyautogui.typewrite(action.text, interval=0.03)
            elif action.action_type == ActionType.KEY:
                pyautogui.hotkey(*action.text.split("+"))
            elif action.action_type == ActionType.SCROLL:
                pyautogui.scroll(int(action.text), x=x, y=y)
            elif action.action_type == ActionType.WAIT:
                await asyncio.sleep(float(action.text) if action.text else 1.0)

            await asyncio.sleep(0.5)
            after = await self.capture()
            return StepResult(action=action, success=True, before_screenshot=before, after_screenshot=after)

        except Exception as e:
            return StepResult(action=action, success=False, error=str(e), before_screenshot=before)


class ScreenAgent:
    """Vision-language agent with Set-of-Mark grounding and reflexion."""

    def __init__(self, vision: VisionGrounding, llm: Any, executor: ActionExecutor):
        self.vision = vision
        self.llm = llm
        self.executor = executor
        self.memory: list[StepResult] = []
        self.max_reflexion_retries = 2

    async def observe(self) -> tuple[list[UIElement], Image.Image]:
        """Capture screen and detect UI elements."""
        screenshot = await self.executor.capture()
        elements = await self.vision.detect_ui(screenshot)
        annotated = self.vision.annotate_set_of_mark(screenshot, elements)
        return elements, annotated

    async def act(self, goal: str, max_steps: int = 10) -> dict:
        """Execute goal through observe-plan-act loop with reflexion."""
        for step in range(max_steps):
            elements, annotated = await self.observe()

            plan = await self.llm.plan_action(
                goal=goal,
                screenshot=annotated,
                elements=[
                    {"id": i + 1, "label": e.label, "type": e.element_type, "text": e.text_content}
                    for i, e in enumerate(elements)
                ],
                history=[
                    {"action": r.action.reasoning, "success": r.success, "error": r.error}
                    for r in self.memory[-5:]
                ],
            )

            if plan.action_type.value == "done":
                return {"status": "success", "steps": step + 1, "history": self._summarize()}

            result = await self.executor.execute(plan, elements)
            self.memory.append(result)

            # Reflexion: verify the action achieved expected state
            if plan.expected_state and not result.success:
                corrected = await self.self_correct(plan, goal, result)
                if not corrected:
                    continue

        return {"status": "max_steps_reached", "steps": max_steps, "history": self._summarize()}

    async def verify_state(self, expected: str) -> bool:
        """Check if current screen matches expected state description."""
        screenshot = await self.executor.capture()
        img_bytes = io.BytesIO()
        screenshot.save(img_bytes, format="PNG")
        img_b64 = base64.b64encode(img_bytes.getvalue()).decode()

        verification = await self.llm.verify(
            image=img_b64,
            expected=expected,
        )
        return verification.matches

    async def self_correct(self, failed_action: Action, goal: str, result: StepResult) -> bool:
        """Attempt to recover from a failed action using reflexion."""
        for retry in range(self.max_reflexion_retries):
            elements, annotated = await self.observe()
            correction = await self.llm.reflect_and_correct(
                goal=goal,
                failed_action=failed_action.reasoning,
                error=result.error,
                screenshot=annotated,
                elements=elements,
            )
            if correction.action_type.value == "done":
                return True
            result = await self.executor.execute(correction, elements)
            if result.success:
                return True
        return False

    def _summarize(self) -> list[dict]:
        return [
            {"step": i + 1, "action": r.action.reasoning, "success": r.success}
            for i, r in enumerate(self.memory)
        ]
