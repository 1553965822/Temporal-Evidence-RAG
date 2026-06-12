#!/usr/bin/env python
from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "raw" / "laws" / "source_docs" / "en"
OUT_DIR = ROOT / "data" / "raw" / "laws" / "en"

USER_PDFS = [
    Path(item)
    for item in os.environ.get("CUADRISK_SOURCE_PDFS", "").split(os.pathsep)
    if item.strip()
]


def rec(
    law_id: str,
    law_key: str,
    law_name: str,
    article_no: str,
    article_text: str,
    article_summary: str,
    risk_tags: list[str],
    risk_categories: list[str],
    valid_from: str,
    valid_to: str | None,
    source_url: str,
    source_file: str,
    status: str | None = None,
    source_type: str = "official_or_free_legal_source",
    version: str | None = None,
) -> dict:
    current_status = status or ("expired" if valid_to else "effective")
    return {
        "law_id": law_id,
        "law_key": law_key,
        "law_name": law_name,
        "article_no": article_no,
        "article_text": article_text,
        "article_summary": article_summary,
        "risk_tags": risk_tags,
        "risk_categories": risk_categories,
        "article_nature": "real_us_legal_section",
        "jurisdiction": "United States",
        "language": "en",
        "version": version or valid_from.replace("-", ""),
        "version_date": valid_from,
        "t_start": valid_from,
        "t_end": valid_to,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "status": current_status,
        "source_url": source_url,
        "source_file": source_file,
        "source_type": source_type,
        "dataset_fit": "CUADRisk",
    }


USC = "https://uscode.house.gov/view.xhtml"
DCCODE = "https://code.dccouncil.gov/us/dc/council/code/sections"
ECFR = "https://www.ecfr.gov/current"

FALLBACK_SOURCE_URLS = {
    "usc15_1.html": "https://www.law.cornell.edu/uscode/text/15/1",
    "usc15_1125.html": "https://www.law.cornell.edu/uscode/text/15/1125",
    "usc15_13.html": "https://www.law.cornell.edu/uscode/text/15/13",
    "usc15_14.html": "https://www.law.cornell.edu/uscode/text/15/14",
    "usc15_45.html": "https://www.law.cornell.edu/uscode/text/15/45",
    "usc15_45b.html": "https://www.law.cornell.edu/uscode/text/15/45b",
    "usc15_77e.html": "https://www.law.cornell.edu/uscode/text/15/77e",
    "usc17_117.html": "https://www.law.cornell.edu/uscode/text/17/117",
    "usc17_204.html": "https://www.law.cornell.edu/uscode/text/17/204",
    "usc18_1839.html": "https://www.law.cornell.edu/uscode/text/18/1839",
    "usc35_261.html": "https://www.law.cornell.edu/uscode/text/35/261",
    "usc9_2.html": "https://www.law.cornell.edu/uscode/text/9/2",
    "cfr16_436_2.html": "https://www.law.cornell.edu/cfr/text/16/436.2",
    "cfr16_436_5.html": "https://www.law.cornell.edu/cfr/text/16/436.5",
    "cfr17_230_506.html": "https://www.law.cornell.edu/cfr/text/17/230.506",
}


def usc_url(title: int, section: str, edition: str = "prelim") -> str:
    req = quote(f"(title:{title} section:{section} edition:{edition})")
    return f"{USC}?req={req}"


def source_name(prefix: str, article: str, suffix: str = "html") -> str:
    clean = (
        f"{prefix}_{article}".replace(" ", "_")
        .replace("§", "sec")
        .replace(":", "-")
        .replace("/", "-")
        .replace(".", "-")
    )
    return f"{clean}.{suffix}"


def build_records() -> list[dict]:
    return [
        rec(
            "USC17_106_PRE1995",
            "USC17_106",
            "Copyright Act, 17 U.S.C.",
            "§106 historical pre-1995",
            "Subject to statutory limitations, the copyright owner had exclusive rights to reproduce, prepare derivative works, distribute copies, and publicly perform or display protected works. This historical version did not yet include the digital audio transmission right later added for sound recordings.",
            "Historical copyright exclusive-rights baseline before the 1995 digital performance amendment.",
            ["License Grant", "Non-Transferable License", "Affiliate License-Licensee", "Affiliate License-Licensor", "Unlimited/All-You-Can-Eat-License", "Source Code Escrow"],
            ["transfer", "generic"],
            "1978-01-01",
            "1995-10-31",
            usc_url(17, "106"),
            "user_title17.pdf",
            source_type="user_supplied_official_pdf_and_uscode_html",
            version="pre_1995",
        ),
        rec(
            "USC17_106_CURRENT",
            "USC17_106",
            "Copyright Act, 17 U.S.C.",
            "§106",
            "Subject to sections 107 through 122, the owner of copyright has the exclusive rights to authorize reproduction, derivative works, distribution, public performance, public display, and, for sound recordings, public performance by digital audio transmission.",
            "Current copyright exclusive rights used to assess whether a license grant, sublicense, software use, source-code access, or affiliate use is overbroad or unsupported.",
            ["License Grant", "Non-Transferable License", "Affiliate License-Licensee", "Affiliate License-Licensor", "Unlimited/All-You-Can-Eat-License", "Source Code Escrow"],
            ["transfer", "generic"],
            "1995-11-01",
            None,
            usc_url(17, "106"),
            "user_title17.pdf",
            source_type="user_supplied_official_pdf_and_uscode_html",
        ),
        rec(
            "USC17_204_CURRENT",
            "USC17_204",
            "Copyright Act, 17 U.S.C.",
            "§204",
            "A transfer of copyright ownership, other than by operation of law, is not valid unless an instrument of conveyance, or a note or memorandum of the transfer, is in writing and signed by the owner of the rights conveyed or by the owner's authorized agent.",
            "Writing requirement for copyright ownership transfers; useful for assignment, exclusive license, and non-transferable license clauses.",
            ["License Grant", "Non-Transferable License", "Anti-Assignment", "Affiliate License-Licensee", "Affiliate License-Licensor", "Source Code Escrow"],
            ["transfer"],
            "1978-01-01",
            None,
            usc_url(17, "204"),
            source_name("usc17", "204"),
        ),
        rec(
            "USC17_117_CURRENT",
            "USC17_117",
            "Copyright Act, 17 U.S.C.",
            "§117",
            "The owner of a copy of a computer program may make or authorize another copy or adaptation of that program only as an essential step in using the program with a machine or for archival purposes, subject to statutory conditions.",
            "Software-copy exception relevant to software license scope, source-code escrow, and operational-use clauses.",
            ["License Grant", "Source Code Escrow", "Unlimited/All-You-Can-Eat-License"],
            ["generic", "transfer"],
            "1980-12-12",
            None,
            usc_url(17, "117"),
            source_name("usc17", "117"),
        ),
        rec(
            "USC35_261_CURRENT",
            "USC35_261",
            "Patent Act, 35 U.S.C.",
            "§261",
            "Patents have the attributes of personal property. Applications for patent, patents, or any interest therein are assignable in law by an instrument in writing, and the owner may grant and convey an exclusive right under the patent or application.",
            "Patent assignment and exclusive-right writing rule for technology license and IP transfer clauses.",
            ["License Grant", "Non-Transferable License", "Affiliate License-Licensee", "Affiliate License-Licensor", "Anti-Assignment"],
            ["transfer"],
            "1952-07-19",
            None,
            usc_url(35, "261"),
            source_name("usc35", "261"),
        ),
        rec(
            "USC18_1836_PRE_DTSA",
            "USC18_1836",
            "Economic Espionage Act, 18 U.S.C.",
            "§1836 historical pre-DTSA",
            "Before the Defend Trade Secrets Act amendments, section 1836 authorized the Attorney General to obtain appropriate injunctive relief against violations of Chapter 90 and gave federal district courts exclusive original jurisdiction for actions under the section.",
            "Expired pre-2016 trade-secret civil-remedy version; included for temporal alignment when old contracts cite federal trade secret remedies before DTSA.",
            ["Source Code Escrow", "Post-Termination Services", "Non-Compete", "Competitive Restriction Exception"],
            ["generic", "term"],
            "1996-10-11",
            "2016-05-10",
            f"{USC}?edition=2010&num=0&req=granuleid%3AUSC-2010-title18-section1836",
            "user_USCODE-2024-title18-partI-chap90-sec1836.pdf",
            source_type="user_supplied_official_pdf_and_uscode_historical_html",
            version="pre_dtsa_2010",
        ),
        rec(
            "USC18_1836_CURRENT",
            "USC18_1836",
            "Defend Trade Secrets Act / Economic Espionage Act, 18 U.S.C.",
            "§1836",
            "An owner of a trade secret that is misappropriated may bring a civil action if the trade secret is related to a product or service used in, or intended for use in, interstate or foreign commerce; the statute also provides for extraordinary civil seizure and other remedies.",
            "Current federal civil action for trade-secret misappropriation, relevant to confidentiality, source-code escrow, transition assistance, and post-termination technology handover.",
            ["Source Code Escrow", "Post-Termination Services", "Non-Compete", "Competitive Restriction Exception"],
            ["generic", "term"],
            "2016-05-11",
            None,
            usc_url(18, "1836"),
            "user_USCODE-2024-title18-partI-chap90-sec1836.pdf",
            source_type="user_supplied_official_pdf_and_uscode_html",
        ),
        rec(
            "USC18_1839_CURRENT",
            "USC18_1839",
            "Economic Espionage Act, 18 U.S.C.",
            "§1839",
            "Section 1839 defines trade secret to include business, technical, scientific, engineering, financial, and other information where the owner has taken reasonable measures to keep it secret and the information derives independent economic value from not being generally known.",
            "Definition of trade secret for confidentiality, source code, know-how transfer, and post-termination assistance clauses.",
            ["Source Code Escrow", "Post-Termination Services", "Non-Compete", "Competitive Restriction Exception"],
            ["generic", "term"],
            "1996-10-11",
            None,
            usc_url(18, "1839"),
            source_name("usc18", "1839"),
        ),
        rec(
            "DC_UCC_2_210_CURRENT",
            "UCC_2_210",
            "District of Columbia Uniform Commercial Code",
            "§28:2-210",
            "A party may delegate performance unless otherwise agreed or unless the other party has a substantial interest in original performance. Rights can generally be assigned unless assignment would materially change the other party's duty, materially increase burden or risk, or impair return performance.",
            "Assignment and delegation rule for anti-assignment, non-transferable license, affiliate license, and change-of-control style clauses.",
            ["Anti-Assignment", "Non-Transferable License", "Rofr/Rofo/Rofn", "Affiliate License-Licensee", "Affiliate License-Licensor"],
            ["transfer"],
            "2005-04-13",
            None,
            f"{DCCODE}/28%3A2-210",
            source_name("dc_ucc", "2_210"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_210_PRE_2005",
            "UCC_2_210",
            "District of Columbia Uniform Commercial Code",
            "§28:2-210 historical pre-2005",
            "The pre-2005 codification supplied the baseline UCC assignment and delegation rule before technical amendments cross-referenced section 28:9-406 and related Article 9 language.",
            "Expired historical assignment/delegation version for temporal comparison with older contracts.",
            ["Anti-Assignment", "Non-Transferable License", "Rofr/Rofo/Rofn"],
            ["transfer"],
            "2001-07-01",
            "2005-04-12",
            f"{DCCODE}/28%3A2-210",
            source_name("dc_ucc", "2_210"),
            source_type="official_dc_code_ucc_codification",
            version="pre_2005",
        ),
        rec(
            "DC_UCC_2_306_CURRENT",
            "UCC_2_306",
            "District of Columbia Uniform Commercial Code",
            "§28:2-306",
            "Output or requirements quantities mean actual output or requirements occurring in good faith, with no unreasonably disproportionate quantity. A lawful exclusive dealing agreement imposes unless otherwise agreed best-efforts duties on the seller to supply and the buyer to promote sales.",
            "Good-faith and best-efforts basis for exclusivity, minimum commitment, volume restriction, and requirements/output clauses.",
            ["Exclusivity", "Minimum Commitment", "Volume Restriction", "Most Favored Nation"],
            ["transfer", "payment"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-306",
            source_name("dc_ucc", "2_306"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_305_CURRENT",
            "UCC_2_305",
            "District of Columbia Uniform Commercial Code",
            "§28:2-305",
            "Parties can conclude a contract for sale even if the price is not settled. In such a case the price is a reasonable price at the time for delivery if the parties so intend or if the price is left to be agreed and they fail to agree.",
            "Open-price rule for fee, revenue sharing, price restriction, and minimum commitment clauses.",
            ["Revenue/Profit Sharing", "Minimum Commitment", "Price Restrictions", "Most Favored Nation"],
            ["payment"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-305",
            source_name("dc_ucc", "2_305"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_313_CURRENT",
            "UCC_2_313",
            "District of Columbia Uniform Commercial Code",
            "§28:2-313",
            "Express warranties arise from affirmations of fact, promises, descriptions, samples, or models that become part of the basis of the bargain. Formal words such as warrant or guarantee are not required.",
            "Express warranty rule for warranty duration and product/service specification clauses.",
            ["Warranty Duration", "License Grant"],
            ["liability"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-313",
            source_name("dc_ucc", "2_313"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_314_CURRENT",
            "UCC_2_314",
            "District of Columbia Uniform Commercial Code",
            "§28:2-314",
            "Unless excluded or modified, a warranty that goods are merchantable is implied in a contract for their sale if the seller is a merchant with respect to goods of that kind; merchantable goods must pass without objection and be fit for ordinary purposes.",
            "Implied warranty of merchantability for warranty duration and product quality clauses.",
            ["Warranty Duration"],
            ["liability"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-314",
            source_name("dc_ucc", "2_314"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_315_CURRENT",
            "UCC_2_315",
            "District of Columbia Uniform Commercial Code",
            "§28:2-315",
            "Where the seller has reason to know a particular purpose for which goods are required and that the buyer relies on the seller's skill or judgment, an implied warranty arises that the goods shall be fit for that purpose unless excluded or modified.",
            "Fitness-for-purpose warranty for service/equipment warranty clauses.",
            ["Warranty Duration"],
            ["liability"],
            "1963-12-30",
            None,
            f"{DCCODE}/28~2-315.html",
            source_name("dc_ucc", "2_315"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_316_CURRENT",
            "UCC_2_316",
            "District of Columbia Uniform Commercial Code",
            "§28:2-316",
            "Words or conduct negating or limiting warranties are construed as consistent with warranty creation where reasonable; to exclude or modify merchantability the language must mention merchantability and, if written, be conspicuous; fitness exclusions must be written and conspicuous.",
            "Warranty disclaimer and limitation rule for warranty duration and liability clauses.",
            ["Warranty Duration", "Uncapped Liability"],
            ["liability"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-316",
            source_name("dc_ucc", "2_316"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_718_CURRENT",
            "UCC_2_718",
            "District of Columbia Uniform Commercial Code",
            "§28:2-718",
            "Damages for breach may be liquidated in the agreement only at an amount reasonable in light of anticipated or actual harm, difficulties of proof, and inconvenience or nonfeasibility of another adequate remedy. Unreasonably large liquidated damages are void as a penalty.",
            "Liquidated damages and penalty-control rule for damages clauses.",
            ["Liquidated Damages", "Uncapped Liability"],
            ["liability"],
            "1963-12-30",
            None,
            f"{DCCODE}/28~2-718.html",
            source_name("dc_ucc", "2_718"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_719_CURRENT",
            "UCC_2_719",
            "District of Columbia Uniform Commercial Code",
            "§28:2-719",
            "The agreement may provide remedies in addition to or in substitution for statutory remedies and may limit or alter recoverable damages, but where an exclusive or limited remedy fails of its essential purpose, remedies under the code may be available; consequential damages may be limited or excluded unless unconscionable.",
            "Limitation-of-remedy and liability-cap evidence for uncapped liability, insurance, and damages clauses.",
            ["Uncapped Liability", "Insurance", "Liquidated Damages"],
            ["liability", "term"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-719",
            source_name("dc_ucc", "2_719"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "DC_UCC_2_725_CURRENT",
            "UCC_2_725",
            "District of Columbia Uniform Commercial Code",
            "§28:2-725",
            "An action for breach of a contract for sale must be commenced within four years after the cause of action accrues. The parties may reduce the period to not less than one year but may not extend it.",
            "Limitations-period rule for warranty duration and post-termination claim-survival clauses.",
            ["Warranty Duration", "Expiration Date", "Post-Termination Services", "Renewal Term"],
            ["liability", "term"],
            "1963-12-30",
            None,
            f"{DCCODE}/28%3A2-725",
            source_name("dc_ucc", "2_725"),
            source_type="official_dc_code_ucc_codification",
        ),
        rec(
            "USC15_1_CURRENT",
            "USC15_1",
            "Sherman Antitrust Act, 15 U.S.C.",
            "§1",
            "Every contract, combination, or conspiracy in restraint of trade or commerce among the several States or with foreign nations is declared illegal and subject to criminal penalties.",
            "Federal antitrust baseline for exclusivity, non-compete, competitive restriction, price restriction, and MFN clauses.",
            ["Exclusivity", "Non-Compete", "Competitive Restriction Exception", "Volume Restriction", "Most Favored Nation", "Price Restrictions"],
            ["transfer", "payment"],
            "1890-07-02",
            None,
            usc_url(15, "1"),
            source_name("usc15", "1"),
        ),
        rec(
            "USC15_14_CURRENT",
            "USC15_14",
            "Clayton Act, 15 U.S.C.",
            "§14",
            "It is unlawful to sell, lease, or contract for sale or lease on the condition, agreement, or understanding that the purchaser or lessee shall not use or deal in competitors' goods where the effect may be to substantially lessen competition or tend to create a monopoly.",
            "Exclusive dealing and tying-style restriction evidence for exclusivity, volume restriction, and non-compete clauses.",
            ["Exclusivity", "Non-Compete", "Competitive Restriction Exception", "Volume Restriction"],
            ["transfer"],
            "1914-10-15",
            None,
            usc_url(15, "14"),
            source_name("usc15", "14"),
        ),
        rec(
            "USC15_13_CURRENT",
            "USC15_13",
            "Robinson-Patman Act / Clayton Act, 15 U.S.C.",
            "§13",
            "It is unlawful for a person engaged in commerce to discriminate in price between different purchasers of commodities of like grade and quality where the effect may substantially lessen competition, tend to create a monopoly, or injure competition.",
            "Price discrimination and competitive pricing evidence for price restriction, MFN, and revenue sharing clauses.",
            ["Price Restrictions", "Most Favored Nation", "Revenue/Profit Sharing"],
            ["payment"],
            "1936-06-19",
            None,
            usc_url(15, "13"),
            source_name("usc15", "13"),
        ),
        rec(
            "USC15_45_CURRENT",
            "USC15_45",
            "Federal Trade Commission Act, 15 U.S.C.",
            "§45",
            "Unfair methods of competition in or affecting commerce, and unfair or deceptive acts or practices in or affecting commerce, are declared unlawful; the Federal Trade Commission is empowered to prevent such practices.",
            "General unfair-competition and unfair-practice evidence for restrictive commercial terms, non-disparagement, franchise, and pricing clauses.",
            ["Non-Compete", "Competitive Restriction Exception", "Most Favored Nation", "Price Restrictions", "Non-Disparagement"],
            ["payment", "generic"],
            "1914-09-26",
            None,
            usc_url(15, "45"),
            source_name("usc15", "45"),
        ),
        rec(
            "USC15_45B_CURRENT",
            "USC15_45B",
            "Consumer Review Fairness Act, 15 U.S.C.",
            "§45b",
            "For covered form contracts, certain terms that prohibit or restrict covered communications such as reviews, performance assessments, or similar analyses, or that penalize such communications, are generally void from the inception of the contract.",
            "Evidence for non-disparagement clauses that restrict reviews or performance assessments in standardized contracts.",
            ["Non-Disparagement"],
            ["generic"],
            "2016-12-14",
            None,
            usc_url(15, "45b"),
            source_name("usc15", "45b"),
        ),
        rec(
            "USC15_1125_PRE_1996",
            "USC15_1125",
            "Lanham Act, 15 U.S.C.",
            "§1125 historical pre-1996",
            "Before the 1996 dilution amendments, subsection (a) focused on false designation of origin, false description, or false representation in commerce creating civil liability for likely commercial injury.",
            "Expired Lanham Act false-designation version for old trademark, brand, and affiliation clauses.",
            ["License Grant", "Parties", "Affiliate License-Licensee", "Affiliate License-Licensor"],
            ["generic", "transfer"],
            "1988-11-16",
            "1996-01-15",
            usc_url(15, "1125"),
            source_name("usc15", "1125"),
            version="pre_1996",
        ),
        rec(
            "USC15_1125_CURRENT",
            "USC15_1125",
            "Lanham Act, 15 U.S.C.",
            "§1125",
            "A person who uses in commerce a word, term, name, symbol, device, false designation of origin, false or misleading description, or false or misleading representation likely to cause confusion as to affiliation, connection, origin, sponsorship, or approval may be liable in a civil action.",
            "Trademark, affiliation, sponsorship, and brand-use evidence for party identity, trademark license, affiliate license, and franchise-like clauses.",
            ["License Grant", "Parties", "Affiliate License-Licensee", "Affiliate License-Licensor"],
            ["generic", "transfer"],
            "1996-01-16",
            None,
            usc_url(15, "1125"),
            source_name("usc15", "1125"),
        ),
        rec(
            "USC15_77E_CURRENT",
            "USC15_77E",
            "Securities Act of 1933, 15 U.S.C.",
            "§77e",
            "Unless a registration statement is in effect, it is unlawful to use interstate commerce or the mails to sell a security through a prospectus or otherwise, or to deliver a security for sale; it is also unlawful to offer to sell or buy a security unless a registration statement has been filed, unless an exemption applies.",
            "Registration baseline for share consideration, revenue/profit sharing, securities issuance, and investment-like consideration clauses.",
            ["Revenue/Profit Sharing", "Minimum Commitment", "Parties"],
            ["payment", "generic"],
            "1933-05-27",
            None,
            usc_url(15, "77e"),
            source_name("usc15", "77e"),
        ),
        rec(
            "CFR17_230_506_CURRENT",
            "CFR17_230_506",
            "Regulation D, 17 C.F.R.",
            "§230.506",
            "Offers and sales satisfying Rule 506(b) or 506(c) are deemed transactions not involving a public offering under Securities Act section 4(a)(2), subject to Regulation D conditions including purchaser limitations, accredited investor verification for 506(c), and related requirements.",
            "Private offering exemption evidence for securities consideration and investment-like contract clauses.",
            ["Revenue/Profit Sharing", "Minimum Commitment", "Parties"],
            ["payment", "generic"],
            "2013-09-23",
            None,
            f"{ECFR}/title-17/chapter-II/part-230/section-230.506",
            source_name("cfr17", "230_506"),
            source_type="official_ecfr",
        ),
        rec(
            "CFR17_230_505_REPEALED",
            "CFR17_230_505",
            "Regulation D, 17 C.F.R.",
            "§230.505 historical repealed",
            "Rule 505 of Regulation D formerly provided a limited-offering exemption but was removed and reserved; the SEC stated that removal of Rule 505 was effective May 22, 2017.",
            "Expired securities offering exemption included for temporal alignment when old contracts or annotations reference Rule 505.",
            ["Revenue/Profit Sharing", "Minimum Commitment", "Parties"],
            ["payment", "generic"],
            "1982-04-15",
            "2017-05-21",
            "https://www.sec.gov/rules-regulations/2016/10/exemptions-facilitate-intrastate-regional-securities-offerings",
            source_name("sec", "rule_505_repeal"),
            source_type="official_sec_rule_release",
            version="repealed_2017",
        ),
        rec(
            "USC9_2_PRE_2022",
            "USC9_2",
            "Federal Arbitration Act, 9 U.S.C.",
            "§2 historical pre-2022",
            "A written arbitration provision in a maritime transaction or contract involving commerce was valid, irrevocable, and enforceable, save upon grounds existing at law or in equity for revocation of any contract.",
            "Expired FAA version before the 2022 chapter 4 carveout amendment.",
            ["Governing Law", "Third Party Beneficiary"],
            ["dispute", "generic"],
            "1947-07-30",
            "2022-03-02",
            f"{USC}?edition=2012&num=0&req=granuleid%3AUSC-2012-title9-section2",
            source_name("usc9", "2"),
            version="pre_2022",
        ),
        rec(
            "USC9_2_CURRENT",
            "USC9_2",
            "Federal Arbitration Act, 9 U.S.C.",
            "§2",
            "A written arbitration provision in a maritime transaction or contract involving commerce is valid, irrevocable, and enforceable, save upon grounds existing at law or in equity for revocation of any contract or as otherwise provided in chapter 4.",
            "Current FAA enforceability rule for arbitration and dispute-resolution clauses.",
            ["Governing Law", "Third Party Beneficiary"],
            ["dispute", "generic"],
            "2022-03-03",
            None,
            usc_url(9, "2"),
            source_name("usc9", "2"),
        ),
        rec(
            "CFR16_436_PRE_2007",
            "CFR16_436",
            "FTC Franchise Rule, 16 C.F.R.",
            "Part 436 historical pre-2007",
            "The original FTC Franchise Rule required pre-sale franchise disclosures and treated covered disclosure failures as unfair or deceptive acts or practices, before the 2007 amended Franchise Rule reorganized and modernized disclosure document requirements.",
            "Expired franchise disclosure regime for contracts whose franchise status is anchored before the 2007 amendments.",
            ["Parties", "License Grant", "Governing Law"],
            ["generic", "dispute"],
            "1979-10-21",
            "2007-06-30",
            "https://www.ftc.gov/legal-library/browse/rules/franchise-rule",
            source_name("ftc", "franchise_rule_old"),
            source_type="official_ftc_rule_page",
            version="pre_2007",
        ),
        rec(
            "CFR16_436_2_CURRENT",
            "CFR16_436",
            "FTC Franchise Rule, 16 C.F.R.",
            "§436.2",
            "In connection with the offer or sale of a franchise located in the United States, unless exempted, it is an unfair or deceptive act or practice to fail to furnish required franchise disclosure documents in the required manner and time.",
            "Current franchise disclosure obligation evidence for clauses denying franchise status or structuring reseller/franchise relationships.",
            ["Parties", "License Grant", "Governing Law"],
            ["generic", "dispute"],
            "2007-07-01",
            None,
            f"{ECFR}/title-16/chapter-I/subchapter-D/part-436/section-436.2",
            source_name("cfr16", "436_2"),
            source_type="official_ecfr",
        ),
        rec(
            "CFR16_436_5_CURRENT",
            "CFR16_436",
            "FTC Franchise Rule, 16 C.F.R.",
            "§436.5",
            "The Franchise Rule specifies disclosure items for the franchise disclosure document, including the franchisor, business experience, litigation, fees, obligations, territory, trademarks, patents, financial performance representations, and related material disclosures.",
            "Franchise disclosure-content evidence for reseller, territory, trademark, fee, and relationship-status clauses.",
            ["Parties", "License Grant", "Revenue/Profit Sharing", "Exclusivity"],
            ["generic", "payment", "transfer"],
            "2007-07-01",
            None,
            f"{ECFR}/title-16/chapter-I/subchapter-D/part-436/section-436.5",
            source_name("cfr16", "436_5"),
            source_type="official_ecfr",
        ),
        rec(
            "COMMONLAW_RESTATEMENT_302",
            "RESTATEMENT_CONTRACTS_302",
            "Restatement (Second) of Contracts",
            "§302 reference",
            "A person may be an intended beneficiary if recognition of a right to performance in the beneficiary is appropriate to effectuate the parties' intention and the circumstances show that the promisee intends to give the beneficiary the benefit of the promised performance.",
            "Non-statutory but standard U.S. contract-law evidence for third-party beneficiary clauses.",
            ["Third Party Beneficiary"],
            ["generic"],
            "1981-01-01",
            None,
            "https://www.law.cornell.edu/wex/third_party_beneficiary",
            source_name("wex", "third_party_beneficiary"),
            source_type="free_legal_reference_summary",
        ),
        rec(
            "COMMONLAW_CONTRACT_FORMATION",
            "COMMONLAW_CONTRACT_FORMATION",
            "U.S. common law contract principles",
            "formation, parties, date, and interpretation",
            "A contract review normally verifies identifiable parties, assent, consideration, effective date, operative document title, amendment integration, and sufficiently definite obligations; ambiguous metadata may affect enforceability and interpretation.",
            "General common-law evidence for parties, document name, agreement date, and effective date fields where no single federal statute governs the clause.",
            ["Parties", "Document Name", "Agreement Date", "Effective Date", "Expiration Date", "Renewal Term"],
            ["generic", "term"],
            "1900-01-01",
            None,
            "https://www.law.cornell.edu/wex/contract",
            source_name("wex", "contract"),
            source_type="free_legal_reference_summary",
        ),
    ]


RISK_TYPE_EXTRA = {
    "Audit Rights": ["COMMONLAW_CONTRACT_FORMATION", "USC15_45_CURRENT"],
    "Insurance": ["DC_UCC_2_719_CURRENT", "COMMONLAW_CONTRACT_FORMATION"],
    "Post-Termination Services": ["USC18_1836_CURRENT", "USC18_1839_CURRENT", "DC_UCC_2_725_CURRENT"],
    "Expiration Date": ["COMMONLAW_CONTRACT_FORMATION", "DC_UCC_2_725_CURRENT"],
    "Renewal Term": ["COMMONLAW_CONTRACT_FORMATION", "DC_UCC_2_725_CURRENT"],
}


def build_mapping(records: list[dict]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for row in records:
        for tag in row["risk_tags"]:
            mapping.setdefault(tag, [])
            if row["law_id"] not in mapping[tag]:
                mapping[tag].append(row["law_id"])
    for risk_type, ids in RISK_TYPE_EXTRA.items():
        mapping.setdefault(risk_type, [])
        for law_id in ids:
            if law_id not in mapping[risk_type]:
                mapping[risk_type].append(law_id)
    for risk_type, ids in mapping.items():
        mapping[risk_type] = ids[:5]
    return dict(sorted(mapping.items()))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(records: list[dict], mapping: dict[str, list[str]]) -> None:
    lines = [
        "# CUADRisk English Legal Validity KB",
        "",
        "This directory contains an English legal knowledge base aligned to the CUADRisk experiment.",
        "",
        "Files:",
        "",
        "- `cuadrisk_legal_validity_kb.en.jsonl`: section-level English legal evidence records.",
        "- `cuadrisk_risk_type_to_law_ids.json`: mapping from CUADRisk `risk_type` to candidate legal evidence IDs.",
        "- `manifest.json`: source-document copy/download log and coverage metadata.",
        "- `cuadrisk_legal_excerpts.md`: human-readable index of the same legal excerpts stored in JSONL.",
        "",
        "The KB intentionally includes historical or repealed records so Temporal-RAG can test legal validity-cycle filtering. Current records use `valid_to: null`; obsolete records have `status: expired` and a concrete `valid_to` date.",
        "",
        f"Record count: {len(records)}",
        f"Mapped CUADRisk risk types: {len(mapping)}",
        "",
        "Historical/obsolete records:",
        "",
    ]
    for row in records:
        if row["status"] == "expired":
            lines.append(f"- `{row['law_id']}`: {row['law_name']} {row['article_no']} ({row['valid_from']} to {row['valid_to']})")
    lines.extend(["", "Risk type coverage:", ""])
    for risk_type, ids in mapping.items():
        lines.append(f"- `{risk_type}`: {', '.join(ids)}")
    (OUT_DIR / "README_CUADRISK_EN_KB.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_excerpts(records: list[dict]) -> None:
    lines = [
        "# CUADRisk English Legal Excerpts",
        "",
        "This file mirrors the `article_text` field from `cuadrisk_legal_validity_kb.en.jsonl` for inspection. It is not used by the experiment runner.",
        "",
    ]
    for row in records:
        lines.extend(
            [
                f"## {row['law_id']}",
                "",
                f"- Law: {row['law_name']}",
                f"- Article: {row['article_no']}",
                f"- Validity: {row['valid_from']} to {row['valid_to'] or 'current'}",
                f"- Status: {row['status']}",
                f"- Source: {row['source_url']}",
                "",
                row["article_text"],
                "",
            ]
        )
    (OUT_DIR / "cuadrisk_legal_excerpts.md").write_text("\n".join(lines), encoding="utf-8")


def copy_user_pdfs() -> list[dict]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in USER_PDFS:
        target = SOURCE_DIR / f"user_{path.name}"
        item = {"source": str(path), "target": str(target), "status": "missing"}
        if path.exists():
            shutil.copy2(path, target)
            item["status"] = "copied"
            item["bytes"] = target.stat().st_size
        copied.append(item)
    return copied


def download_sources(records: list[dict]) -> list[dict]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    for row in records:
        url = row["source_url"]
        filename = row["source_file"]
        if filename.startswith("user_"):
            continue
        seen[url] = filename
    logs = []
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("User-Agent", "paper-aligned-contract-rag/1.0 educational legal KB builder"),
        ("Accept", "text/html,application/pdf,*/*"),
    ]
    for url, filename in sorted(seen.items(), key=lambda x: x[1]):
        target = SOURCE_DIR / filename
        status = "downloaded"
        error = ""
        try:
            with opener.open(url, timeout=8) as response:
                content = response.read()
            lowered = content[:2000].lower()
            if b"request access" in lowered or b"access denied" in lowered:
                raise RuntimeError("remote site returned an access/request page instead of legal text")
            target.write_bytes(content)
            time.sleep(0.2)
        except Exception as exc:  # keep KB build usable even when a public site blocks scripted fetches
            status = "download_failed"
            error = str(exc)
            if target.exists():
                target.unlink()
            fallback = FALLBACK_SOURCE_URLS.get(filename)
            if fallback:
                try:
                    with opener.open(fallback, timeout=12) as response:
                        content = response.read()
                    target.write_bytes(content)
                    status = "fallback_downloaded"
                    error = f"official fetch failed: {error}; fallback: {fallback}"
                    time.sleep(0.2)
                except Exception as fallback_exc:
                    error = f"{error}; fallback failed: {fallback_exc}"
        logs.append({"url": url, "target": str(target), "status": status, "error": error})
    return logs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = build_records()
    mapping = build_mapping(records)
    copied = copy_user_pdfs()
    download_log = download_sources(records)

    write_jsonl(OUT_DIR / "cuadrisk_legal_validity_kb.en.jsonl", records)
    (OUT_DIR / "cuadrisk_risk_type_to_law_ids.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(records, mapping)
    write_excerpts(records)
    manifest = {
        "dataset_fit": "CUADRisk",
        "record_count": len(records),
        "effective_records": sum(1 for r in records if r["status"] == "effective"),
        "expired_records": sum(1 for r in records if r["status"] == "expired"),
        "covered_risk_types": sorted(mapping),
        "source_docs_dir": str(SOURCE_DIR),
        "kb_path": str(OUT_DIR / "cuadrisk_legal_validity_kb.en.jsonl"),
        "mapping_path": str(OUT_DIR / "cuadrisk_risk_type_to_law_ids.json"),
        "readme_path": str(OUT_DIR / "README_CUADRISK_EN_KB.md"),
        "excerpts_path": str(OUT_DIR / "cuadrisk_legal_excerpts.md"),
        "copied_user_pdfs": copied,
        "download_log": download_log,
        "note": "English CUADRisk legal KB. Historical/expired records are intentionally included for Temporal-RAG validity-cycle testing.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
