from openai import OpenAI
import json

client = OpenAI()


class InterviewAgent:
    def __init__(self, jd, resume_text):
        self.jd = jd
        self.resume = resume_text
        self.history = []
        self.scores = []

    def next_turn(self, candidate_answer=None):

        if candidate_answer:
            self.history.append(
                {"role": "user", "content": candidate_answer}
            )
            self._evaluate_answer(candidate_answer)

        system_prompt = f"""
You are a professional AI interviewer.

Job Description:
{self.jd}

Candidate Resume:
{self.resume}

Rules:
- Ask ONE question at a time
- Ask follow-up questions when useful
- Gradually increase difficulty
- Keep questions concise
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                *self.history
            ]
        )

        question = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": question})
        return question

    # --------------------------------------------------
    # SCORING ENGINE
    # --------------------------------------------------
    def _evaluate_answer(self, answer):
        eval_prompt = f"""
Evaluate the candidate's answer using the rubric below.
Return ONLY valid JSON.

Answer:
{answer}

Rubric (0–10):
- technical
- relevance
- communication
- confidence

Respond format:
{{
  "technical": number,
  "relevance": number,
  "communication": number,
  "confidence": number,
  "summary": "one sentence evaluation"
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": eval_prompt}
            ]
        )

        try:
            data = json.loads(response.choices[0].message.content)
            self.scores.append(data)
        except Exception:
            pass  # fail silently to avoid breaking interview

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------
    def final_result(self):
        if not self.scores:
            return None

        avg = lambda k: sum(s[k] for s in self.scores) / len(self.scores)

        result = {
            "technical": round(avg("technical"), 2),
            "relevance": round(avg("relevance"), 2),
            "communication": round(avg("communication"), 2),
            "confidence": round(avg("confidence"), 2),
        }

        overall = round(sum(result.values()) / 4, 2)

        if overall >= 7.5:
            verdict = "HIRE"
        elif overall >= 5.0:
            verdict = "MAYBE"
        else:
            verdict = "REJECT"

        return {
            "overall": overall,
            "verdict": verdict,
            "breakdown": result
        }

    # Compatibility wrapper
    def next_question(self, candidate_answer=None):
        return self.next_turn(candidate_answer)
