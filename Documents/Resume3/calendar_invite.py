# calendar_invite.py
import os
import uuid
import smtplib
import re
from email.message import EmailMessage
from email.utils import formatdate
from datetime import datetime, timedelta, timezone

# Read SMTP config from environment variables (secure)
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASS = os.environ.get("SMTP_PASS")
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))

EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"


def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(re.fullmatch(EMAIL_REGEX, email))


def _to_utc_z(dt: datetime) -> str:
    """
    Return UTC time string in format YYYYMMDDTHHMMSSZ
    Accepts naive or aware datetimes.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y%m%dT%H%M%SZ")


def generate_ics(event_title: str,
                 start_dt: datetime,
                 end_dt: datetime,
                 description: str,
                 meeting_link: str,
                 organizer_email: str,
                 attendee_email: str) -> str:
    """
    Create a standards-compliant ICS content string for a meeting request.
    Uses UTC times (with trailing Z).
    """
    uid = f"{uuid.uuid4()}@ai-screening"
    dtstamp = _to_utc_z(datetime.utcnow())
    dtstart = _to_utc_z(start_dt)
    dtend = _to_utc_z(end_dt)

    ics_lines = [
        "BEGIN:VCALENDAR",
        "PRODID:-//AI Screening//EN",
        "VERSION:2.0",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{event_title}",
        f"DESCRIPTION:{description}\\nMeeting Link: {meeting_link}",
        f"ORGANIZER:mailto:{organizer_email}",
        f"ATTENDEE;CN=Candidate;RSVP=TRUE:mailto:{attendee_email}",
        "SEQUENCE:0",
        "STATUS:CONFIRMED",
        "TRANSP:OPAQUE",
        "END:VEVENT",
        "END:VCALENDAR",
    ]
    return "\r\n".join(ics_lines)


def send_calendar_invite(to_email: str,
                         event_start: datetime,
                         duration_minutes: int = 30,
                         meeting_link: str = "",
                         organizer_email: str = None) -> (bool, str):
    """
    Send an ICS calendar invite via SMTP to to_email.
    Uses SMTP credentials from environment variables.
    Returns (ok: bool, message: str).
    """
    # Basic validation
    if not is_valid_email(to_email):
        return False, f"Invalid recipient email: {to_email}"

    if not SMTP_USER or not SMTP_PASS:
        return False, "SMTP credentials not configured. Set SMTP_USER and SMTP_PASS environment variables."

    # Determine organizer
    organizer = organizer_email or SMTP_USER

    # prepare start & end datetimes (preserve tz if provided, otherwise assume local sent as UTC)
    start_dt = event_start
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    # generate ICS content
    ics_content = generate_ics(
        event_title="AI Screening Interview",
        start_dt=start_dt,
        end_dt=end_dt,
        description="Your AI Screening Interview has been scheduled.",
        meeting_link=meeting_link or "",
        organizer_email=organizer,
        attendee_email=to_email
    )

    try:
        # Build email
        msg = EmailMessage()
        msg["Subject"] = "AI Screening Interview Invitation"
        msg["From"] = organizer
        msg["To"] = to_email
        msg["Date"] = formatdate(localtime=True)
        human_text = f"Your interview is scheduled for {start_dt.isoformat()}.\nOpen the attached calendar file to add this event to your calendar."
        msg.set_content(human_text)

        # Attach ICS bytes and then set helpful headers on that part so mail apps treat it as a meeting request
        ics_bytes = ics_content.encode("utf-8")
        msg.add_attachment(ics_bytes, maintype="text", subtype="calendar", filename="invite.ics")
        # The new attachment becomes the last payload item; add helpful headers
        try:
            part = msg.get_payload()[-1]
            # Some mail clients respect these headers
            part.add_header("Content-Class", "urn:content-classes:calendarmessage")
            part.add_header("Method", "REQUEST")
            # ensure content-type indicates method
            part.replace_header("Content-Type", 'text/calendar; method=REQUEST; charset="UTF-8"')
        except Exception:
            # best-effort; not fatal
            pass

        # Send via SMTP SSL
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)

        return True, f"Invite sent to {to_email}"
    except Exception as e:
        return False, f"Error sending invite to {to_email}: {e}"
