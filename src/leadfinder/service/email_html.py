"""The branded email: an approved email's plain text set under the seller's logo, as it is sent.

Emails are written, checked and approved as plain text (outreach.py, writer.py); this wraps one for sending, the way
AuxoCase's email_html.py does. Email clients ignore stylesheets, so everything is tables and inline styles. The
palette is the one the pages use (base.html's tokens): a near-black letterhead with the logo and the name in gold, a
white sheet for the words on the site's grey, a line naming what the product does, and a muted footer.

The logo is static/clearwell-logo.png: a gold water drop with a check in it (a clear well; an appointment
confirmed). In the email it travels inside the message and is named by its content id (LOGO_CID; mailer.py attaches
it), so no mail client has to fetch it from anywhere; the lead's page shows the same picture from /assets/.

The footer the guardrail requires (the postal address and the way to opt out, CAN-SPAM) is the part of the email
after its "--" line; it is set under the sheet, in full, so the branded email keeps it.
"""
from __future__ import annotations

import re
from html import escape
from pathlib import Path

from ..outreach import SELLER

BRAND = SELLER
TAGLINE = "Scheduling for independent practices"
FEATURES = ("Online booking", "Reminders", "Waitlist")       # what outreach.py says the product is, short enough for one line

LOGO_FILE = Path(__file__).resolve().parent / "static" / "clearwell-logo.png"
LOGO_CID = "clearwell-logo"             # the logo's name inside a sent message
LOGO_URL = "/assets/clearwell-logo.png"  # and on the service's own pages

BG = "#f3f2f2"          # the page behind the sheet
CARD = "#ffffff"
LINE = "#e4e0e0"
INK = "#201f1d"
MUTED = "#6b6767"
BAND = "#1a1919"        # the letterhead
BAND_MUTED = "#aeaba3"
GOLD = "#b68235"        # the accent: the rule under the letterhead
GOLD_LIGHT = "#e1ad66"  # the name on the letterhead
GOLD_DEEP = "#7d5411"   # links and small capitals on white

HEADING = "'Cormorant Garamond', Georgia, 'Times New Roman', serif"
BODY = "Lora, Georgia, 'Times New Roman', serif"
CAPS = "font-size:11px;line-height:16px;letter-spacing:0.16em;text-transform:uppercase"

URL = re.compile(r"https?://[^\s<>\"]+")
SIGNATURE = "\n--\n"    # what follows is the footer (outreach.FOOTER)


def _line(text: str) -> str:
    """One line of the email as HTML: escaped, with any link made clickable."""
    out, at = [], 0
    for m in URL.finditer(text):
        out.append(escape(text[at:m.start()]))
        url = m.group(0).rstrip(".,;:)")
        tail = m.group(0)[len(url):]
        out.append(f'<a href="{escape(url, quote=True)}" style="color:{GOLD_DEEP};text-decoration:underline">{escape(url)}</a>{escape(tail)}')
        at = m.end()
    out.append(escape(text[at:]))
    return "".join(out)


def _paragraph(block: str, first: bool = False) -> str:
    lines = block.split("\n")
    if first and len(lines) == 1 and len(block) <= 48 and block.rstrip().endswith(","):
        # the greeting ("Hi Kim,") stands a little taller, in the heading face
        return f'<p style="margin:0 0 18px 0;font-family:{HEADING};font-size:24px;line-height:30px;font-weight:600;color:{INK}">{_line(block)}</p>'
    return (f'<p style="margin:0 0 18px 0;font-family:{BODY};font-size:16px;line-height:1.7;color:{INK}">'
            f'{"<br>".join(_line(ln) for ln in lines)}</p>')


def branded(body: str, sender: str = "", logo: str = f"cid:{LOGO_CID}") -> str:
    """The email's body as a branded HTML email. `sender` is the SDR's name for the footer, when known. `logo` is
    where the picture comes from: the message itself by default, LOGO_URL when a page of the service shows it."""
    words, _, footer = body.strip().partition(SIGNATURE)
    blocks = [b for b in re.split(r"\n\s*\n", words.strip()) if b.strip()]
    foot = [_line(ln) for ln in footer.strip().split("\n") if ln.strip()]
    if sender:
        foot.append(f"Sent by {escape(sender)} for {escape(BRAND)}")
    dot = f'<span style="color:{GOLD}">&nbsp;&nbsp;&middot;&nbsp;&nbsp;</span>'
    return (
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%;background:{BG};margin:0;padding:0">'
        f'<tr><td align="center" style="padding:32px 12px">'
        f'<table role="presentation" width="600" cellpadding="0" cellspacing="0" border="0" style="max-width:600px;width:100%;margin:0 auto">'
        # the letterhead: the logo, then the name and what the seller does
        f'<tr><td style="background:{BAND};padding:24px 36px 22px 36px;border-radius:10px 10px 0 0">'
        f'<table role="presentation" cellpadding="0" cellspacing="0" border="0"><tr>'
        f'<td valign="middle" style="padding:0;vertical-align:middle">'
        f'<img src="{escape(logo, quote=True)}" width="48" height="48" alt="{escape(BRAND)} logo" '
        f'style="display:block;width:48px;height:48px;border:0;outline:none"></td>'
        f'<td valign="middle" style="padding:0 0 0 14px;vertical-align:middle">'
        f'<span style="font-family:{HEADING};font-size:30px;line-height:32px;font-weight:600;letter-spacing:0.01em;color:{GOLD_LIGHT}">{escape(BRAND)}</span><br>'
        f'<span style="font-family:{BODY};{CAPS};color:{BAND_MUTED}">{escape(TAGLINE)}</span>'
        f'</td></tr></table>'
        f'</td></tr>'
        f'<tr><td style="background:{GOLD};height:3px;line-height:3px;font-size:3px;padding:0">&nbsp;</td></tr>'
        # the words
        f'<tr><td style="background:{CARD};border-left:1px solid {LINE};border-right:1px solid {LINE};padding:36px 36px 10px 36px">'
        f'{"".join(_paragraph(b, first=i == 0) for i, b in enumerate(blocks))}'
        f'</td></tr>'
        # what the product does, under a hairline at the foot of the sheet
        f'<tr><td style="background:{CARD};border:1px solid {LINE};border-top:none;padding:0 36px 26px 36px;border-radius:0 0 10px 10px">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="width:100%"><tr>'
        f'<td align="center" style="border-top:1px solid {LINE};padding:18px 0 0 0;text-align:center;font-family:{BODY};{CAPS};color:{GOLD_DEEP}">'
        f'{dot.join(escape(f) for f in FEATURES)}'
        f'</td></tr></table>'
        f'</td></tr>'
        # the footer: the postal address and the way to opt out, then who sent it
        f'<tr><td style="padding:20px 16px 0 16px;font-family:{BODY};font-size:12px;line-height:18px;color:{MUTED};text-align:center">'
        f'{"<br>".join(foot)}'
        f'</td></tr>'
        f'</table></td></tr></table>'
    )
