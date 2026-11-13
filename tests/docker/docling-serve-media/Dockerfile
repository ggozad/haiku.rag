# docling-serve with audio and video support, for the integration tests.
# Adds ffmpeg and the unreleased ASR settings of docling-project/docling-jobkit#286
# and docling-project/docling-serve#762. Not published.
FROM mwader/static-ffmpeg:7.1.1@sha256:11a44711684c0b9f754c047dcd64235b8b52deab251bd0e0a86f22faa160749c AS ffmpeg

FROM quay.io/docling-project/docling-serve:v1.38.2

COPY --from=ffmpeg /ffmpeg /ffprobe /usr/local/bin/

ARG JOBKIT_SHA=111ca1985b1b06adb48348c91b5d4a43eb945373
ARG SERVE_SHA=2f6c81e19dfc955e82eb832efec4d8734aa82194

RUN pip install --no-cache-dir --no-deps --force-reinstall \
    "docling-jobkit @ https://github.com/chiptoe-svg/docling-jobkit/archive/${JOBKIT_SHA}.tar.gz"

# The image runs docling-serve from its source checkout in the working directory,
# which shadows anything installed into site-packages.
USER 0
RUN curl -fsSL "https://github.com/chiptoe-svg/docling-serve/archive/${SERVE_SHA}.tar.gz" \
    | tar -xz -C /tmp \
    && rm -rf /opt/app-root/src/docling_serve \
    && cp -r "/tmp/docling-serve-${SERVE_SHA}/docling_serve" /opt/app-root/src/docling_serve \
    && rm -rf "/tmp/docling-serve-${SERVE_SHA}"
USER 1001
