// From https://issues.chromium.org/issues/40052254

sk_sp Make(const SkGlyphRunList& glyphRunList,  
                                   GrStrikeCache\* strikeCache,  
                                   const SkMatrix& drawMatrix,  
                                   GrColor color,  
                                   bool forceWForDistanceFields) {  
  
    static_assert(sizeof(ARGB2DVertex) <= sizeof(Mask2DVertex));  
    static_assert(alignof(ARGB2DVertex) <= alignof(Mask2DVertex));  
    size_t quadSize = sizeof(Mask2DVertex) \* kVerticesPerGlyph;  
    if (drawMatrix.hasPerspective() || forceWForDistanceFields) {  
        static_assert(sizeof(ARGB3DVertex) <= sizeof(SDFT3DVertex));  
        static_assert(alignof(ARGB3DVertex) <= alignof(SDFT3DVertex));  
        quadSize = sizeof(SDFT3DVertex) \* kVerticesPerGlyph;  
    }  
  
    // We can use the alignment of SDFT3DVertex as a proxy for all Vertex alignments.  
    static_assert(alignof(SDFT3DVertex) >= alignof(Mask2DVertex));  
    // Assume there is no padding needed between glyph pointers and vertices.  
    static_assert(alignof(GrGlyph\*) >= alignof(SDFT3DVertex));  
  
    // In the arena, the layout is GrGlyph\*... | SDFT3DVertex... | SubRun, so there is no padding  
    // between GrGlyph\* and SDFT3DVertex, but padding is needed between the Mask2DVertex array  
    // and the SubRun.  
    size_t vertexToSubRunPadding = alignof(SDFT3DVertex) - alignof(SubRun);  
    size_t arenaSize =  
            sizeof(GrGlyph\*) \* glyphRunList.totalGlyphCount()  
          + quadSize \* glyphRunList.totalGlyphCount()  
          + glyphRunList.runCount() \* (sizeof(SubRun) + vertexToSubRunPadding); 
  
    size_t allocationSize = sizeof(GrTextBlob) + arenaSize; 
  
    void\* allocation = ::operator new (allocationSize);  
  
    SkColor initialLuminance = SkPaintPriv::ComputeLuminanceColor(glyphRunList.paint());  
    sk_sp<GrTextBlob> blob{new (allocation) GrTextBlob{  
            arenaSize, strikeCache, drawMatrix, glyphRunList.origin(),  
            color, initialLuminance, forceWForDistanceFields}};  
  
    return blob;  
}  
