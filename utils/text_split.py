def slpit_into_chunk(content,chunk_size=800, overlap=100):
    print("chunk started.....................")
    chunk= []

    start=0
    while start < len(content):
        print("chunking..........................")

        end = start + chunk_size
        chunk.append(content[start:end])
        start += chunk_size

        start += chunk_size - overlap
    
    print('chunk completed!')
    return chunk
