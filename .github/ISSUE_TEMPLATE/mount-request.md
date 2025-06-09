name: 🛠️ Camera Mount Request
description: Request a new camera mount design for a specific 3D printer.
title: "[Mount Request] Printer Model: "
labels: [enhancement, hardware]
assignees: ''

body:
  - type: input
    id: printer-model
    attributes:
      label: Printer Make and Model
      description: Specify the full model name of the printer (e.g., Creality Ender 3 v1).
      placeholder: e.g., "Creality Ender 3 v1"
    validations:
      required: true

  - type: dropdown
    id: firmware
    attributes:
      label: Firmware Type
      description: Which firmware is the printer running?
      options:
        - Marlin
        - Klipper
        - Other / Unknown
    validations:
      required: true

  - type: input
    id: build-volume
    attributes:
      label: Build Volume
      placeholder: e.g., "250 × 210 × 210 mm"
    validations:
      required: false

  - type: textarea
    id: dimensions
    attributes:
      label: Bed and Carriage Dimensions
      description: Include bed size, carriage width, and any relevant measurements.
    validations:
      required: false

  - type: textarea
    id: photos
    attributes:
      label: Photos / Drawings of the Print Head Carriage
      description: Upload clear photos with the print head removed. Include ruler or measurements if possible.
    validations:
      required: false

  - type: input
    id: technical-resources
    attributes:
      label: Technical Drawings or CAD Files (optional)
      placeholder: Paste any links to STLs, DXFs, or mechanical drawings. This will increase the chances of getting a model made.
    validations:
      required: false

  - type: textarea
    id: other-notes
    attributes:
      label: Additional Notes
      description: Any other information we should know?
    validations:
      required: false