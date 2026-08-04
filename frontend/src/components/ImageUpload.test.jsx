import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ImageUpload from "./ImageUpload";

function createFile(name, type, size = 1024) {
  const blob = new Blob(["a".repeat(size)], { type });
  return new File([blob], name, { type });
}

function renderUpload(overrides = {}) {
  const props = {
    image: null,
    setImage: vi.fn(),
    imagePreview: null,
    setImagePreview: vi.fn(),
    ...overrides,
  };
  return render(<ImageUpload {...props} />);
}

describe("ImageUpload", () => {
  it("renders placeholder when no image is selected", () => {
    renderUpload();
    expect(screen.getByText(/attach an image/i)).toBeInTheDocument();
  });

  it("renders the file input with accepted extensions", () => {
    renderUpload();
    const input = screen.getByLabelText(/attach an image/i);
    expect(input).toHaveAttribute("type", "file");
    expect(input).toHaveAttribute("accept", ".png,.jpg,.jpeg,.webp");
  });

  it("shows file info and remove button when image is selected", () => {
    renderUpload({
      image: createFile("test.png", "image/png"),
      imagePreview: "blob:http://localhost/test",
    });
    expect(screen.getByText("test.png")).toBeInTheDocument();
    expect(screen.getByText("Remove")).toBeInTheDocument();
  });

  it("calls setImage and setImagePreview on valid file selection", () => {
    const setImage = vi.fn();
    const setImagePreview = vi.fn();
    const { container } = render(
      <ImageUpload
        image={null}
        setImage={setImage}
        imagePreview={null}
        setImagePreview={setImagePreview}
      />
    );

    const file = createFile("photo.png", "image/png");
    const input = container.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [file] } });

    expect(setImage).toHaveBeenCalledWith(file);
    expect(setImagePreview).toHaveBeenCalledWith(expect.stringContaining("blob:"));
  });

  it("rejects unsupported file type", () => {
    const setImage = vi.fn();
    const setImagePreview = vi.fn();
    const { container } = render(
      <ImageUpload
        image={null}
        setImage={setImage}
        imagePreview={null}
        setImagePreview={setImagePreview}
      />
    );

    const file = createFile("doc.pdf", "application/pdf");
    const input = container.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [file] } });

    expect(setImage).toHaveBeenCalledWith(null);
    expect(setImagePreview).toHaveBeenCalledWith(null);
  });

  it("rejects file exceeding max size", () => {
    const setImage = vi.fn();
    const setImagePreview = vi.fn();
    const { container } = render(
      <ImageUpload
        image={null}
        setImage={setImage}
        imagePreview={null}
        setImagePreview={setImagePreview}
      />
    );

    const large = createFile("large.png", "image/png", 6 * 1024 * 1024);
    const input = container.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [large] } });

    expect(setImage).toHaveBeenCalledWith(null);
    expect(setImagePreview).toHaveBeenCalledWith(null);
  });

  it("calls handleRemove and clears state", () => {
    const setImage = vi.fn();
    const setImagePreview = vi.fn();
    render(
      <ImageUpload
        image={createFile("test.jpg", "image/jpeg")}
        setImage={setImage}
        imagePreview="blob:http://localhost/preview"
        setImagePreview={setImagePreview}
      />
    );

    fireEvent.click(screen.getByText("Remove"));

    expect(setImage).toHaveBeenCalledWith(null);
    expect(setImagePreview).toHaveBeenCalledWith(null);
  });

  it("does nothing when no file is selected", () => {
    const setImage = vi.fn();
    const setImagePreview = vi.fn();
    const { container } = render(
      <ImageUpload
        image={null}
        setImage={setImage}
        imagePreview={null}
        setImagePreview={setImagePreview}
      />
    );

    const input = container.querySelector('input[type="file"]');
    fireEvent.change(input, { target: { files: [] } });

    expect(setImage).not.toHaveBeenCalled();
    expect(setImagePreview).not.toHaveBeenCalled();
  });
});
