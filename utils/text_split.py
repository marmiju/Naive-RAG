def slpit_into_chunk(content,chunk_size=200, overlap=10):
    print("chunk started.....................")
    chunks= []

    start=0
    while start < len(content):
        print("chunking..........................")

        end = start + chunk_size
        chunks.append(content[start:end])
        start += chunk_size

        start += chunk_size - overlap
    
    print('chunk completed!')
    return chunks
