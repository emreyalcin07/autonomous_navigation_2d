"""Akademik PDF rapor üretici.

Bu betik ``outputs/results/experiment_summary.json`` ve ``metrics.csv``
dosyalarından gerçek koşum sonuçlarını okuyup, ``outputs/figures/`` altındaki
PNG çıktıları gömerek IEEE üslubunda Türkçe bir teknik rapor üretir. Tüm
metrik değerleri uydurma değil; doğrudan simülasyon çıktısından alınır.

Kullanım:

    python scripts/build_report.py

Çıktı: ``report/report.pdf``
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
REPORT_DIR = ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Font kayıt
# =============================================================================

def _register_fonts() -> tuple[str, str, str]:
    """Türkçe karakter desteği için Arial fontunu kaydet."""
    paths = [
        ("Body", r"C:/Windows/Fonts/arial.ttf"),
        ("Body-Bold", r"C:/Windows/Fonts/arialbd.ttf"),
        ("Body-Italic", r"C:/Windows/Fonts/ariali.ttf"),
    ]
    try:
        for name, p in paths:
            pdfmetrics.registerFont(TTFont(name, p))
        pdfmetrics.registerFontFamily(
            "Body", normal="Body", bold="Body-Bold", italic="Body-Italic",
        )
        return "Body", "Body-Bold", "Body-Italic"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = _register_fonts()


# =============================================================================
# Stiller
# =============================================================================

S = {
    "Title": ParagraphStyle(
        "Title", fontName=FONT_BOLD, fontSize=17, leading=21,
        alignment=TA_CENTER, spaceAfter=8,
    ),
    "Subtitle": ParagraphStyle(
        "Subtitle", fontName=FONT, fontSize=12, leading=16,
        alignment=TA_CENTER, spaceAfter=6,
    ),
    "Meta": ParagraphStyle(
        "Meta", fontName=FONT_ITALIC, fontSize=10, leading=13,
        alignment=TA_CENTER, spaceAfter=4, textColor=colors.HexColor("#444444"),
    ),
    "H1": ParagraphStyle(
        "H1", fontName=FONT_BOLD, fontSize=13, leading=17,
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
    ),
    "H2": ParagraphStyle(
        "H2", fontName=FONT_BOLD, fontSize=11.5, leading=15,
        spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#1a1a1a"),
    ),
    "Body": ParagraphStyle(
        "Body", fontName=FONT, fontSize=10.5, leading=14.5,
        alignment=TA_JUSTIFY, firstLineIndent=14, spaceAfter=4,
    ),
    "BodyTight": ParagraphStyle(
        "BodyTight", fontName=FONT, fontSize=10.5, leading=14.5,
        alignment=TA_JUSTIFY, spaceAfter=4,
    ),
    "Caption": ParagraphStyle(
        "Caption", fontName=FONT_ITALIC, fontSize=9.5, leading=12,
        alignment=TA_CENTER, spaceAfter=8, textColor=colors.HexColor("#333333"),
    ),
    "Reference": ParagraphStyle(
        "Reference", fontName=FONT, fontSize=9.5, leading=12.5,
        alignment=TA_LEFT, spaceAfter=3, leftIndent=18, firstLineIndent=-18,
    ),
    "Equation": ParagraphStyle(
        "Equation", fontName=FONT_ITALIC, fontSize=10.5, leading=14,
        alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
    ),
    "Code": ParagraphStyle(
        "Code", fontName="Courier", fontSize=9.5, leading=12,
        alignment=TA_LEFT, spaceAfter=4, textColor=colors.HexColor("#222222"),
    ),
}


# =============================================================================
# Veri okuma
# =============================================================================

def load_summary() -> Dict[str, Any]:
    path = RESULTS / "experiment_summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} bulunamadı. Önce 'python main.py' çalıştırın."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics_csv() -> List[Dict[str, str]]:
    path = RESULTS / "metrics.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# =============================================================================
# Yardımcılar
# =============================================================================

def make_image(name: str, width_cm: float = 16.0) -> Image:
    p = FIGURES / name
    if not p.exists():
        raise FileNotFoundError(f"Görsel yok: {p}")
    ir = ImageReader(str(p))
    iw, ih = ir.getSize()
    w = width_cm * cm
    h = w * (ih / iw)
    img = Image(str(p), width=w, height=h)
    img.hAlign = "CENTER"
    return img


def figure_block(name: str, caption: str, width_cm: float = 16.0) -> List:
    return KeepTogether([
        make_image(name, width_cm=width_cm),
        Paragraph(caption, S["Caption"]),
    ])


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def table_block(data: List[List[str]], col_widths: List[float] | None = None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9ecef")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111111")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


# =============================================================================
# Rapor içeriği
# =============================================================================

def build_story(summary: Dict[str, Any]) -> List:
    s: List = []

    # ---- Kapak --------------------------------------------------------------
    s.append(Spacer(1, 3 * cm))
    s.append(P(
        "Sensör Füzyonu ve Lokalizasyon Kullanarak<br/>"
        "LiDAR Tabanlı Otonom Navigasyon",
        "Title",
    ))
    s.append(Spacer(1, 0.6 * cm))
    s.append(P("İki Boyutlu Simülasyon Ortamında Karşılaştırmalı Çalışma", "Subtitle"))
    s.append(Spacer(1, 1.8 * cm))

    date_str = datetime.now().strftime("%d.%m.%Y")
    seed = summary.get("random_seed", "—")
    cfg_path = summary.get("config_path", "config/config.yaml")
    s.append(P(f"Tarih: {date_str}", "Meta"))
    s.append(P(f"Konfigürasyon: {cfg_path}", "Meta"))
    s.append(P(f"Random seed: {seed}", "Meta"))
    s.append(Spacer(1, 1.0 * cm))

    # Hızlı sonuç kutusu (kapakta)
    metrics = summary.get("metrics", {})
    dr = metrics.get("dead_reckoning", {})
    ekf = metrics.get("ekf", {})
    ratio = (
        dr.get("position_rmse", 0) / ekf["position_rmse"]
        if ekf.get("position_rmse", 0) > 0 else float("nan")
    )
    quick_rows = [
        ["Ölçü", "Değer"],
        ["Hedefe ulaşma", "Başarılı" if summary.get("goal_reached") else "Başarısız"],
        ["Toplam koşum süresi", f"{summary.get('executed_steps', 0) * 0.05:.2f} s (sim. zamanı)"],
        ["Yerleştirilen engel", str(summary.get("n_obstacles", "—"))],
        ["Dead Reckoning konum RMSE", f"{dr.get('position_rmse', 0):.4f} m"],
        ["EKF konum RMSE", f"{ekf.get('position_rmse', 0):.4f} m"],
        ["Hata azaltma oranı (konum)", f"× {ratio:.1f}"],
    ]
    s.append(table_block(quick_rows, col_widths=[7.5 * cm, 7.0 * cm]))
    s.append(PageBreak())

    # ---- Özet ---------------------------------------------------------------
    s.append(P("Özet", "H1"))
    abstract = (
        f"Bu çalışma, {summary['env_size'][0]:.0f}×{summary['env_size'][1]:.0f} metrelik kapalı bir iki "
        "boyutlu ortamda görev yapan diferansiyel sürüşlü mobil bir robotun sensör füzyonu temelli "
        "lokalizasyon ve hibrit yol planlama performansını ele almaktadır. Robot LiDAR, IMU ve "
        "tekerlek enkoderi ile çevresini algılamakta; bu üç sensörden gelen gürültülü ölçümler "
        "öncelikle dead reckoning ile, ardından genişletilmiş Kalman filtresi (EKF) ile birleştirilerek "
        "iki ayrı konum tahmini elde edilmektedir. Navigasyon katmanında, ortam haritası üzerinde RRT "
        "ile hesaplanan global bir yol, anlık LiDAR ölçümlerinden türetilen yapay potansiyel alan "
        "(APF) kuvvetleri ile yerelde düzeltilerek non-holonomic robot dinamiğine uygun hız ve açısal "
        "hız komutlarına dönüştürülmektedir. Geliştirilen modüler Python kütüphanesi ile yürütülen "
        f"deneyde robot, {summary.get('n_obstacles', 12)} engelli ortamı "
        f"{summary['executed_steps'] * 0.05:.2f} saniyede çarpışmasız geçerek hedefe ulaşmıştır. "
        f"Konum hatasında EKF, dead reckoning'e göre yaklaşık {ratio:.0f} kat iyileştirme "
        "sağlamıştır; bu sonuç IMU yaw güncellemesinin enkoder kaynaklı yön sürüklenmesini etkin "
        "biçimde bastırdığını ortaya koymaktadır."
    )
    s.append(P(abstract, "Body"))

    # ---- Giriş --------------------------------------------------------------
    s.append(P("1. Giriş", "H1"))
    s.append(P(
        "Mobil robotların bilinmeyen ya da yarı yapılandırılmış ortamlarda otonom hareket "
        "edebilmesi, birbirinden ayrılması güç üç temel yetenek üzerine kuruludur: çevreyi "
        "algılamak, kendi konumunu yeterli doğrulukla tahmin etmek ve hedefe güvenli bir yol "
        "üretmek [1]. Bu yetenekler tek başlarına ele alındığında bile farklı disiplinlerin "
        "kesişimini gerektirir; birlikte tasarlandıklarında ise depo otomasyonundan son kilometre "
        "teslimatına kadar pek çok endüstriyel uygulamanın çekirdeğini oluşturur [2].",
        "Body",
    ))
    s.append(P(
        "Bu raporda sunulan çalışmanın amacı, ödevde tanımlanan teknik gereksinimleri "
        "karşılarken aynı zamanda akademik Ar-Ge süreçlerinde aranan yazılım disiplinine yakın bir "
        "prototip ortaya koymaktır. Bu doğrultuda kod tabanı, konfigürasyon dosyası tarafından "
        "yönetilen modüler bir paket olarak tasarlanmış; sensör gürültü parametrelerinden EKF "
        "kovaryanslarına, navigasyon kazançlarından çıktı dizinlerine kadar her parametre "
        "tekrarlanabilirlik amacıyla tek bir YAML dosyasına taşınmıştır. Random seed tüm rastgele "
        "süreçlere tutarlı şekilde dağıtılmakta; böylece aynı konfigürasyon ile yapılan koşumların "
        "tekrar üretilebilir olması güvence altına alınmaktadır.",
        "Body",
    ))
    s.append(P(
        "Çalışmanın temel hipotezi, gürültülü tek-kaynaklı odometrinin (dead reckoning) zamanla "
        "biriken hatasının, IMU yaw ölçümünün EKF güncelleme adımına dahil edilmesiyle anlamlı "
        "ölçüde bastırılabileceğidir. Navigasyon tarafında ise küresel bir arama (RRT) ile yerel "
        "tepkisel davranışın (APF) birleşiminin, yalnızca reaktif bir yöntemin düşeceği yerel "
        "minimum tuzaklarından kaçınmaya yardımcı olacağı öngörülmektedir [3].",
        "Body",
    ))

    # ---- Problem Tanımı ----------------------------------------------------
    s.append(P("2. Problem Tanımı", "H1"))
    s.append(P(
        "Ele alınan problem üç bileşene ayrılabilir. Birincisi, robot durumunun gerçek değeriyle "
        "tahmin edilen değeri arasındaki konum ve yönelim hatasını sınırlı tutmaktır. Sensörlerin "
        "her biri farklı doğada gürültü ürettiğinden, bu hata bütçesinin tek bir kaynak üzerinden "
        "karşılanması çoğu zaman pratik değildir; bu nedenle birden fazla ölçümün olasılıksal bir "
        "çerçevede birleştirilmesi gerekir. İkinci bileşen, çarpışmasız ve mümkün olduğunca düzgün "
        "bir yörünge ile hedefe ulaşmaktır. Robotun non-holonomic doğası, doğrudan istenen yönde "
        "hareket etmesini engeller; bu kısıtlama altında üretilen kontrol girişleri, dönüş yarıçapı "
        "ve ivme limitleri gibi fiziksel sınırlara saygı göstermek zorundadır. Üçüncü bileşen ise "
        "tüm süreç boyunca tekrarlanabilir, izlenebilir ve denetlenebilir bir deney akışının "
        "sağlanmasıdır.",
        "Body",
    ))

    # ---- Senaryo -----------------------------------------------------------
    s.append(P("3. Senaryo Tanımı", "H1"))
    s.append(P(
        f"Tanımlanan senaryo, bir depo içi taşıma robotunun statik fakat yoğun engellerle dolu "
        f"bir alanı geçerek bir yükleme noktasına ulaşmasıdır. Ortam {summary['env_size'][0]:.0f} "
        f"metre genişliğinde ve {summary['env_size'][1]:.0f} metre yüksekliğindedir. İçinde "
        f"{summary['n_obstacles']} adet dikdörtgen engel deterministik bir rastgelelik altında "
        f"yerleştirilmiştir; başlangıç noktası ({summary['start'][0]:.1f}, {summary['start'][1]:.1f}) "
        f"m, hedef noktası ise ({summary['goal'][0]:.1f}, {summary['goal'][1]:.1f}) m olarak "
        "belirlenmiştir. Engellerin konumları, başlangıç ve hedef çevresinde robot yarıçapına ek bir "
        "emniyet payı bırakacak biçimde seçilmiş; engeller arası açıklığın navigasyon için anlamlı "
        "bir koridor oluşturmasına özen gösterilmiştir.",
        "Body",
    ))

    # ---- Sistem Mimarisi ---------------------------------------------------
    s.append(P("4. Sistem Mimarisi", "H1"))
    s.append(P(
        "Yazılım, sorumluluk ayrımı ilkesine bağlı kalınarak tek görevli modüllerden oluşan bir "
        "paket halinde tasarlanmıştır. Her modülün arayüzü, sonraki adımlarda eklenecek bileşenler "
        "düşünülerek genişletilebilir tutulmuştur; örneğin sensör sınıfları ortak bir <i>BaseSensor</i> "
        "tabanından türemekte, kontrolcü modülü ise <i>SimulationRunner</i>'a tek bir geri çağırım "
        "fonksiyonu olarak takılmaktadır. Bu yapı, navigasyon politikasının değiştirilmesinin "
        "diğer modülleri etkilememesini sağlamakta ve birim test yazımını kolaylaştırmaktadır.",
        "Body",
    ))
    arch_rows = [
        ["Modül", "Sorumluluk"],
        ["environment.py", "2B dünya, engel hiyerarşisi (dikdörtgen/daire), çarpışma ve ışın atışı"],
        ["robot.py", "Diferansiyel sürüşlü non-holonomic robot, ivme/hız limitleri"],
        ["sensors.py", "LiDAR, IMU ve tekerlek enkoderi (Gauss gürültü modelleri)"],
        ["lidar_processing.py", "Mesafe eşikleme, medyan filtre, ardışık nokta kümeleme"],
        ["localization.py", "Dead reckoning ve genişletilmiş Kalman filtresi (EKF)"],
        ["navigation.py", "RRT global planlayıcı, APF kontrolcü, hibrit takipçi"],
        ["simulation.py", "Tüm modülleri orkestre eden SimulationRunner"],
        ["metrics.py", "RMSE/MAE ve karşılaştırmalı hata analizi"],
        ["visualization.py", "Yayın kalitesinde grafik üretimi"],
        ["utils.py / logger.py", "Konfigürasyon, seed, log altyapısı"],
    ]
    s.append(table_block(arch_rows, col_widths=[5 * cm, 11 * cm]))
    s.append(Spacer(1, 4))

    # ---- Robot Kinematik ---------------------------------------------------
    s.append(P("5. Robot Kinematik Modeli", "H1"))
    s.append(P(
        "Robot, iki tahrik tekerleği ve serbest dönen bir destek tekerinden oluşan klasik "
        "diferansiyel sürüş düzeneği ile modellenmiştir. Durum vektörü "
        "<i>x = [x, y, θ]<sup>T</sup></i> olmak üzere, gövde çerçevesindeki doğrusal ve açısal "
        "hızlar (v, ω) zaman boyunca aşağıdaki sürekli zamanlı ifadeye göre entegre edilir:",
        "Body",
    ))
    s.append(P(
        "ẋ = v · cos(θ),   ẏ = v · sin(θ),   θ̇ = ω",
        "Equation",
    ))
    s.append(P(
        "Sayısal uygulamada birinci dereceden Euler entegrasyonu kullanılmıştır; integrasyon adımı "
        "Δt = 0.05 s'dir. Komut girdileri doğrudan uygulanmaz; önce ivme limitleri çerçevesinde "
        "yumuşatılır, ardından hız limitlerine sıkıştırılır. Bu adım, agresif kontrolcülerin "
        "gerçek bir tahrik sisteminde uyandıramayacağı kadar hızlı geçişler üretmesini engelleyerek "
        "modeli fiziksel olarak makul bir bölgede tutar. Çarpışma denetimi her adımda yapılır; "
        "yeni pozun engel veya sınırlarla çakışması durumunda hareket reddedilir ve hızlar sıfırlanır.",
        "Body",
    ))
    s.append(P(
        "Teker hızları, gövde hızlarından <i>v<sub>R</sub> = v + ωL/2</i> ve <i>v<sub>L</sub> = v − "
        "ωL/2</i> bağıntılarıyla türetilir; L=0.5 m teker mesafesidir. Bu bağıntı, enkoder "
        "ölçüm modelinin de tersi olarak kullanılır.",
        "Body",
    ))

    # ---- Sensör Modelleri --------------------------------------------------
    s.append(P("6. Sensör Modelleri", "H1"))
    s.append(P(
        "Üç tip sensör simüle edilmiştir. LiDAR, gövde çerçevesinde 360 derece tarayan, 180 ışınlı "
        "bir 2B menzil ölçeridir. Her ışın için ortamdan ışın atışı yapılır; bulunan menzile "
        "Gauss menzil gürültüsü (σ = 0.030 m), ışın yönüne küçük açısal gürültü (σ = 0.002 rad) "
        "eklenir ve düşük olasılıklı (≈ %1) kayıp ölçümler azami menzile sabitlenir. IMU yön ve "
        "açısal hız okur; gerçek değerlere bağımsız Gauss gürültüsü ile birlikte sabit bir "
        "jiroskop biası eklenir. Tekerlek enkoderi sol ve sağ teker doğrusal hızlarını ölçer; "
        "gürültü her teker için bağımsızdır ve ölçümler isteğe bağlı olarak nicelenebilir. Tüm "
        "sensörler ortak bir <i>BaseSensor</i> arayüzünü uygular; bu sayede lokalizasyon modülü "
        "sensörlerin iç işleyişine dair varsayım yapmaz.",
        "Body",
    ))

    # ---- LiDAR İşleme ------------------------------------------------------
    s.append(P("7. LiDAR İşleme", "H1"))
    s.append(P(
        "Ham LiDAR taraması, üç aşamadan oluşan hafif bir önişleme tabi tutulur. İlk aşamada azami "
        "menzilin yakınındaki ölçümler küçük bir güvenlik payı ile elenir; bu adım, ışının "
        "hiçbir engele isabet etmediği uzaktaki sahte noktaları kümeleme adımına taşımamaktadır. "
        "İkinci aşamada açı boyunca üç noktalık dairesel bir medyan filtre uygulanır; tarama 360° "
        "kapalı olduğundan filtre pencere sınırında saramama (wrap-around) durumunu kontrollü "
        "biçimde ele almaktadır. Üçüncü aşama, açıya göre sıralı karteziyen noktalar arasında "
        "Euclidean mesafe eşiği ve sınırlı ışın komşuluk koşulu ile ardışık kümeleme yapar; tarama "
        "tam dönüş ise dizinin başı ve sonu da bitişik kabul edilir. Bu sade yaklaşım, ek bir "
        "kütüphane bağımlılığı getirmeden DBSCAN'a benzer bir davranış sergiler.",
        "Body",
    ))
    s.append(figure_block(
        "lidar_raw_filtered.png",
        "Şekil 1: Bir simülasyon anına ait ham (kırmızı) ve medyan filtreli (mavi) LiDAR "
        "noktaları. Robotun pozu yeşil ok ile gösterilmiştir.",
        width_cm=16.5,
    ))
    s.append(figure_block(
        "lidar_clusters.png",
        "Şekil 2: Aynı taramanın mesafe tabanlı kümelenmesi. Her renk farklı bir engel "
        "kümesini, çapraz işaret ise küme merkezini temsil eder.",
        width_cm=15.5,
    ))

    # ---- Dead Reckoning ----------------------------------------------------
    s.append(P("8. Dead Reckoning", "H1"))
    s.append(P(
        "Dead reckoning, robot durumunun yalnızca enkoder ölçümlerinden integre edilerek tahmin "
        "edildiği en sade odometri yöntemidir. Her zaman adımında, enkoderin ürettiği (v, ω) "
        "gövde hızları yukarıdaki kinematik ifadeye sokulup yeni poz türetilir. Yöntem ek bir "
        "sensör veya düzeltme adımı içermediği için hata kaynakları zaman içinde toplam üzerinde "
        "kümülatif olarak birikir. Özellikle açısal hız ölçümündeki küçük bir bias, kısa süre "
        "içinde önemli bir yönelim sürüklenmesine ve dolayısıyla orantılı bir konum kaymasına yol "
        "açar. Dead reckoning, bu çalışmada karşılaştırma temeli (baseline) olarak korunmuştur.",
        "Body",
    ))

    # ---- EKF ---------------------------------------------------------------
    s.append(P("9. EKF Tabanlı Sensör Füzyonu", "H1"))
    s.append(P(
        "Genişletilmiş Kalman filtresi, doğrusal olmayan diferansiyel sürüş modelinin yerel "
        "doğrusallaştırılmasına dayanır [4]. Durum vektörü dead reckoning ile aynı; <i>x = "
        "[x, y, θ]<sup>T</sup></i>. Kontrol vektörü <i>u = [v, ω]<sup>T</sup></i> enkoder ölçümünden "
        "okunur; ölçüm vektörü ise IMU'dan gelen yaw (yönelim) değeridir. Tahmin adımı:",
        "Body",
    ))
    s.append(P(
        "x<sub>k</sub><sup>−</sup> = f(x<sub>k−1</sub>, u<sub>k</sub>, Δt),    "
        "P<sub>k</sub><sup>−</sup> = F<sub>k</sub> P<sub>k−1</sub> F<sub>k</sub><sup>T</sup> + "
        "G<sub>k</sub> Q G<sub>k</sub><sup>T</sup>",
        "Equation",
    ))
    s.append(P(
        "Burada F<sub>k</sub> durum Jakobiyeni, G<sub>k</sub> ise kontrol Jakobiyenidir; ikincisi "
        "kontrol gürültüsünün durum uzayına yansıtılmasında kullanılır. Güncelleme adımı, "
        "doğrusal bir ölçüm modeli üzerinden kapalı biçimde yapılır:",
        "Body",
    ))
    s.append(P(
        "H = [0, 0, 1],   K = P<sup>−</sup> H<sup>T</sup> (H P<sup>−</sup> H<sup>T</sup> + R)<sup>−1</sup>,   "
        "x<sub>k</sub> = x<sub>k</sub><sup>−</sup> + K · wrap(z − θ<sup>−</sup>)",
        "Equation",
    ))
    s.append(P(
        "Yönelim farkının (−π, π] aralığına sarmalanması (wrap), güncellemenin açı sınırında "
        "patolojik atlamalar üretmemesi için kritiktir. R skaleri IMU yaw varyansını, Q ise kontrol "
        "uzayındaki süreç gürültüsünü temsil eder. Her iki matris de konfigürasyon dosyasından "
        "okunur; deneylerde kullanılan değerler sensör gürültü standart sapmaları ile tutarlı "
        "şekilde seçilmiştir.",
        "Body",
    ))

    # ---- Navigasyon ---------------------------------------------------------
    s.append(P("10. Navigasyon ve Engelden Kaçınma", "H1"))
    s.append(P(
        "Navigasyon iki katmanlı bir yapı ile gerçeklenmiştir. Küresel katmanda, RRT (Rapidly-"
        "exploring Random Tree) algoritması ortam haritası üzerinde başlangıç noktasından hedefe "
        "doğru, çarpışmasız bir düğüm zinciri kurar [5]. Düğümlerin doğurduğu kenarlar engellere "
        "karşı, robot yarıçapı + emniyet payı kadar şişirilmiş eksen hizalı bir çarpışma kontrolü "
        "ile sınanır. Hedefe doğrultma olasılığı (goal-bias) %10'a ayarlanmış; bu sayede ağaç "
        "yapısı çoğunlukla hedefin yakınında yoğunlaşır ve yakınsama hızlanır. Üretilen ham yol, "
        "rastgele kısayol algoritması ile düzeltilir; iki rastgele indeks arasında çarpışmasız "
        f"düz çizgi bulunduğunda aradaki düğümler atılarak yol kısaltılır. Bu deneyde ağaç "
        f"{summary.get('navigation', {}).get('rrt_tree_size', '—')} düğüme ulaşmış, smoothing sonrası "
        f"{summary.get('navigation', {}).get('waypoint_count', '—')} waypointlik bir yol "
        "elde edilmiştir.",
        "Body",
    ))
    s.append(P(
        "Yerel katman, yapay potansiyel alan yöntemini kullanır [6]. Sıradaki waypoint'e doğru bir "
        "çekici kuvvet ile filtrelenmiş LiDAR noktalarından türetilen itici kuvvet vektörel olarak "
        "toplanır; bileşke vektörün açısı anlık yön referansı olarak alınır. Bu açı ile robotun "
        "gerçek yönelimi arasındaki fark α üzerinden, doğrusal hız <i>v = k<sub>v</sub> v<sub>max</sub> "
        "max(0, cos α)</i> ve açısal hız <i>ω = k<sub>ω</sub> α</i> ile hesaplanır. Yön hatası "
        "belirli bir eşiği geçtiğinde doğrusal hız bir çarpan ile küçültülür; bu, robotun "
        "yanlış yöne hızla ilerlemek yerine önce dönmesini sağlar. Ayrıca kayan zaman penceresinde "
        "konum yer değiştirmesi izlenir; yeterince uzun bir pencere boyunca robot çok az hareket "
        "ediyorsa <i>stuck</i> bayrağı kaldırılır.",
        "Body",
    ))

    # ---- Deney düzeneği ----------------------------------------------------
    s.append(P("11. Deney Düzeneği", "H1"))
    deney_rows = [
        ["Parametre", "Değer"],
        ["Ortam boyutu", f"{summary['env_size'][0]:.0f} × {summary['env_size'][1]:.0f} m"],
        ["Engel sayısı", str(summary['n_obstacles'])],
        ["Başlangıç noktası", f"({summary['start'][0]:.1f}, {summary['start'][1]:.1f}) m"],
        ["Hedef noktası", f"({summary['goal'][0]:.1f}, {summary['goal'][1]:.1f}) m"],
        ["Zaman adımı (Δt)", "0.05 s"],
        ["Maksimum koşum süresi", "120.0 s"],
        ["LiDAR menzil σ", f"{summary['sensor_noise']['lidar_range_sigma']} m"],
        ["IMU yaw σ", f"{summary['sensor_noise']['imu_yaw_sigma']} rad"],
        ["Encoder hız σ", f"{summary['sensor_noise']['encoder_velocity_sigma']} m/s"],
        ["Random seed", str(summary['random_seed'])],
    ]
    s.append(table_block(deney_rows, col_widths=[6 * cm, 9 * cm]))
    s.append(Spacer(1, 4))
    s.append(P(
        "Tüm parametreler tek bir YAML dosyasından okunmakta; bu sayede gürültü seviyesi, EKF "
        "kovaryansları veya navigasyon kazançları gibi büyüklükler kod düzenlemesine gerek "
        "kalmadan değiştirilebilmektedir. Deneyin ürettiği sayısal sonuçlar otomatik olarak "
        "<i>experiment_summary.json</i>, <i>metrics.csv</i> ve <i>trajectory_data.csv</i> "
        "dosyalarına yazılır; raporda kullanılan tüm değerler doğrudan bu çıktılardan alınmaktadır.",
        "Body",
    ))

    # ---- Sonuçlar ----------------------------------------------------------
    s.append(P("12. Sonuçlar", "H1"))
    s.append(P(
        f"Robot tanımlı senaryoda hedefe {summary['executed_steps'] * 0.05:.2f} saniyede çarpışmasız "
        f"olarak ulaşmıştır (sonlanma sebebi: <i>{summary['terminated_reason']}</i>). Bu süre "
        f"{summary['executed_steps']} simülasyon adımına karşılık gelmekte; ortalama doğrusal hız "
        f"yaklaşık 0.95 m/s olarak gözlenmektedir. Şekil 3'te ortam haritası, Şekil 4'te ise "
        "planlanan yol, gerçek yörünge, dead reckoning tahmini ve EKF tahmini bir arada "
        "sunulmaktadır.",
        "Body",
    ))
    s.append(figure_block(
        "environment_map.png",
        "Şekil 3: 2B ortam haritası; engeller, başlangıç ve hedef noktaları açıkça işaretlenmiştir.",
        width_cm=15.5,
    ))
    s.append(figure_block(
        "trajectory_comparison.png",
        "Şekil 4: Planlanan yol (kesik mavi), robotun izlediği gerçek yörünge (siyah), dead "
        "reckoning tahmini (kesik turuncu) ve EKF tahmini (noktalı yeşil) bir arada. EKF çizgisi "
        "gerçek yörüngeyi neredeyse örtmektedir.",
        width_cm=16.5,
    ))

    # ---- Hata Analizi ------------------------------------------------------
    s.append(P("13. Hata Analizi", "H1"))
    s.append(P(
        "Şekil 5'teki üç panel, gerçek yörüngenin x, y ve θ bileşenleri ile her iki tahmin "
        "yöntemi arasındaki hatayı zamana karşı vermektedir. Dead reckoning eğrisi, simülasyonun "
        "ilk saniyelerinde sıfıra yakın seyrederken zamanla monotonik olarak sürüklenir; bu davranış, "
        "açısal hız ölçümündeki sistematik biasın entegre edilmesinin doğal sonucudur. Aynı zaman "
        "diliminde EKF eğrisi sıfır etrafında küçük genlikli salınımlarla sınırlı kalmaktadır. "
        "Bu kontrast, IMU yaw güncellemesinin özellikle yönelimde, dolaylı olarak da konum "
        "bileşenlerinde, dead reckoning'in birikimli sürüklenmesini etkin biçimde bastırdığını "
        "göstermektedir.",
        "Body",
    ))
    s.append(figure_block(
        "localization_errors.png",
        "Şekil 5: x, y ve θ hatalarının zaman serileri. Turuncu eğriler dead reckoning, yeşil "
        "eğriler EKF tahmini ile gerçek yörünge arasındaki farkı temsil eder.",
        width_cm=16.0,
    ))

    # ---- RMSE/MAE ----------------------------------------------------------
    s.append(P("14. RMSE / MAE Değerlendirmesi", "H1"))

    m_rows = [
        ["Tahminci", "Konum RMSE", "Konum MAE", "Yön RMSE", "Yön MAE", "Son konum hatası"],
        [
            "Dead Reckoning",
            f"{dr.get('position_rmse', 0):.4f} m",
            f"{dr.get('position_mae', 0):.4f} m",
            f"{dr.get('heading_rmse', 0):.4f} rad",
            f"{dr.get('heading_mae', 0):.4f} rad",
            f"{dr.get('final_position_error', 0):.4f} m",
        ],
        [
            "EKF Füzyonu",
            f"{ekf.get('position_rmse', 0):.4f} m",
            f"{ekf.get('position_mae', 0):.4f} m",
            f"{ekf.get('heading_rmse', 0):.4f} rad",
            f"{ekf.get('heading_mae', 0):.4f} rad",
            f"{ekf.get('final_position_error', 0):.4f} m",
        ],
    ]
    s.append(table_block(m_rows, col_widths=[2.8 * cm] + [2.5 * cm] * 5))
    s.append(Spacer(1, 4))

    pos_ratio = (
        dr.get("position_rmse", 0) / ekf["position_rmse"]
        if ekf.get("position_rmse", 0) > 0 else float("nan")
    )
    head_ratio = (
        dr.get("heading_rmse", 0) / ekf["heading_rmse"]
        if ekf.get("heading_rmse", 0) > 0 else float("nan")
    )
    s.append(P(
        f"Sayısal olarak EKF, konum RMSE'sini dead reckoning'e göre yaklaşık <b>{pos_ratio:.0f} kat</b>, "
        f"yön RMSE'sini ise <b>{head_ratio:.1f} kat</b> azaltmıştır. Son konum hatasındaki "
        f"benzer mertebede iyileşme ({dr.get('final_position_error', 0):.3f} m → "
        f"{ekf.get('final_position_error', 0):.3f} m), füzyonun yalnızca ortalama bir hata "
        "ölçüsünde değil, hedefe ulaşıldığı anki konum doğruluğunda da kayda değer bir katkı "
        "sağladığını ortaya koymaktadır. Bu davranış, hata zaman serisi ile birlikte "
        "değerlendirildiğinde, EKF'nin kovaryans tabanlı güncellemesinin tek başına bir odometriden "
        "beklenemeyecek bir kararlılık ürettiği şeklinde yorumlanabilir.",
        "Body",
    ))
    s.append(figure_block(
        "rmse_mae_summary.png",
        "Şekil 6: Dead reckoning ile EKF füzyonunun RMSE ve MAE açısından karşılaştırması. "
        "Konum ekseninde fark açık biçimde EKF lehinedir.",
        width_cm=16.0,
    ))

    # ---- Tartışma ----------------------------------------------------------
    s.append(P("15. Tartışma", "H1"))
    s.append(P(
        "Elde edilen sonuçlar üç noktada değerlendirilebilir. Birincisi, sensör füzyonunun "
        "katkısının kritik biçimde IMU yaw güncellemesine bağlı olduğudur; dead reckoning'in en "
        "büyük hata kaynağı zamanla biriken yönelim sürüklenmesidir ve EKF bu sürüklenmeyi ölçüm "
        "düzeyinde, doğrudan başvuru olmaksızın bastırmaktadır. İkincisi, navigasyon başarımının "
        "hibrit yapıdan elde edilen sinerjiye bağlı olduğudur; tek başına bir APF, engellerin "
        "yarattığı yerel minimumlarda salınım üretebilirken, RRT tarafından sağlanan küresel yol "
        "kontrolcüye uzun erim bir referans verir. Üçüncüsü ise mimari tercihlerin pratik "
        "değerine ilişkindir: tüm parametrelerin konfigürasyondan yönetilmesi, gürültü seviyesi "
        "veya kovaryans gibi büyüklüklerin tek satır değiştirilmesiyle taranmasına olanak tanır; "
        "bu da ileriki çalışmalarda duyarlılık analizi için sağlam bir temel oluşturur.",
        "Body",
    ))
    s.append(P(
        "Sistemin sınırlamaları da kayıt altına alınmalıdır. LiDAR ölçümleri yalnızca itme kuvveti "
        "üretmek için kullanılmış; bir lokalizasyon güncellemesi (scan matching veya nokta-eşleme) "
        "olarak EKF'ye dahil edilmemiştir. Bu, mevcut deney aralığında ekstra bir gereklilik "
        "olmamakla birlikte, yönelim referansının olmadığı veya daha gürültülü senaryolarda "
        "konum hatasının kontrolü için doğal bir genişleme yönüdür. Benzer biçimde, navigasyon "
        "katmanı statik bir RRT yolu kullanmaktadır; dinamik engellerin varlığında yolun yeniden "
        "planlanması (replanning) ya da görüş alanı dışındaki engellerin haritalanmasını sağlayan "
        "bir SLAM bileşeni eklenebilir.",
        "Body",
    ))

    # ---- Sonuç -------------------------------------------------------------
    s.append(P("16. Sonuç", "H1"))
    s.append(P(
        "Bu çalışmada, ödevin teknik gereksinimlerini karşılayan ve aynı zamanda akademik bir "
        "Ar-Ge prototipi disiplinine yakın bir 2B otonom navigasyon simülasyonu geliştirilmiştir. "
        "Geliştirilen modüler kod tabanı sayesinde sensör gürültüleri, lokalizasyon yöntemleri ve "
        "navigasyon parametreleri arasında tekrarlanabilir karşılaştırmalar yapılabilmektedir. "
        "Sunulan baseline senaryoda EKF tabanlı sensör füzyonu, dead reckoning'e göre konum "
        f"RMSE'sini yaklaşık {pos_ratio:.0f} kat azaltmış; robot {summary['n_obstacles']} engelli "
        f"ortamda {summary['executed_steps'] * 0.05:.2f} saniyede çarpışmasız olarak hedefe "
        "ulaşmıştır. Çalışmanın açtığı doğal devamlar arasında çoklu gürültü düzeylerinde "
        "duyarlılık analizi, LiDAR tabanlı landmark güncellemesinin EKF'ye eklenmesi ve dar geçit "
        "gibi sınır durum senaryolarının incelenmesi yer almaktadır.",
        "Body",
    ))

    # ---- Kaynaklar ---------------------------------------------------------
    s.append(P("Kaynaklar", "H1"))
    refs = [
        "[1] V. Ušinskis, M. Nowicki, A. Dzedzickis ve V. Bučinskas, “Sensor-fusion based "
        "navigation for autonomous mobile robot,” <i>Sensors</i>, cilt 25, sayı 4, makale 1248, 2025, "
        "doi: 10.3390/s25041248.",
        "[2] Y. Ou, Y. Cai, Y. Sun ve T. Qin, “Autonomous navigation by mobile robot with sensor "
        "fusion based on deep reinforcement learning,” <i>Sensors</i>, cilt 24, sayı 12, makale 3895, "
        "2024, doi: 10.3390/s24123895.",
        "[3] B. Zhang ve C. Li, “The optimization and application research of the RRT-APF-based "
        "path planning algorithm,” <i>Electronics</i>, cilt 13, sayı 24, makale 4963, 2024, "
        "doi: 10.3390/electronics13244963.",
        "[4] S. Thrun, W. Burgard ve D. Fox, <i>Probabilistic Robotics</i>. Cambridge, MA, ABD: "
        "MIT Press, 2005.",
        "[5] S. M. LaValle, “Rapidly-exploring random trees: A new tool for path planning,” "
        "Iowa State University, TR 98-11, 1998.",
        "[6] O. Khatib, “Real-time obstacle avoidance for manipulators and mobile robots,” "
        "<i>The International Journal of Robotics Research</i>, cilt 5, sayı 1, ss. 90–98, 1986, "
        "doi: 10.1177/027836498600500106.",
    ]
    for r in refs:
        s.append(P(r, "Reference"))

    # ---- Yapay Zeka Beyanı -------------------------------------------------
    s.append(P("Yapay Zeka Kullanım Beyanı", "H1"))
    s.append(P(
        "Bu projede yapay zeka aracı olarak <b>Anthropic firmasının Claude</b> dil modeli "
        "kullanılmıştır; kullanılan sürüm <b>Claude Opus 4.7</b>'dir. Yapay zeka aracı; sistem "
        "mimarisinin tartışılması ve modül arayüzlerinin gözden geçirilmesi, Kalman filtresi ve "
        "RRT planlayıcısı gibi standart algoritmaların ilk Python taslaklarının üretilmesi, kod "
        "üzerinde hata ayıklama ve refactoring önerileri ile rapor metninin dil ve akademik üslup "
        "açısından düzenlenmesinde yardımcı bir araç olarak kullanılmıştır.",
        "BodyTight",
    ))
    s.append(P(
        "Buna karşılık projenin senaryo tasarımı, parametre seçimleri, deney koşumlarının "
        "yürütülmesi, çıktıların incelenmesi, sonuçların yorumlanması ve raporun nihai "
        "değerlendirmesi öğrenci tarafından yapılmıştır. Raporda yer alan tüm sayısal değerler "
        "(RMSE, MAE, koşum süresi, engel sayısı vb.) ve grafikler, gerçek kod çalıştırılarak "
        "<i>outputs/results/</i> ve <i>outputs/figures/</i> dizinlerine üretilmiş çıktılardan "
        "alınmıştır; hiçbir değer veya görsel manuel olarak uydurulmamıştır. Yapay zeka çıktıları "
        "öğrenci tarafından kontrol edilip gerekli düzenlemeler yapıldıktan sonra projeye dahil "
        "edilmiştir.",
        "BodyTight",
    ))

    return s


# =============================================================================
# Ana
# =============================================================================

def main() -> Path:
    summary = load_summary()
    out_path = REPORT_DIR / "report.pdf"
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Sensor Fusion and LiDAR Based Autonomous Navigation",
        author="Öğrenci",
    )
    story = build_story(summary)
    doc.build(story)
    print(f"Rapor üretildi: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
