#!/usr/bin/env python3
"""Debug script to analyze why ceteris paribus is failing"""

from cv_templates import CVTemplates
import hashlib

# Generate CVs for two different names with same inputs
variables1 = {'name': 'Brad', 'university': 'State University', 'experience': '2', 'address': '123 Main St, Anytown, USA'}
variables2 = {'name': 'Todd', 'university': 'State University', 'experience': '2', 'address': '123 Main St, Anytown, USA'}

cv1 = CVTemplates.generate_cv_content('software_engineer', variables1, 'borderline')
cv2 = CVTemplates.generate_cv_content('software_engineer', variables2, 'borderline')

print('=== CV 1 (Brad) ===')
print(cv1[:500])
print('...')
print()
print('=== CV 2 (Todd) ===')
print(cv2[:500])
print('...')
print()

# Show the difference in seeds
seed1 = int(hashlib.md5(f'Brad_borderline_software_engineer'.encode()).hexdigest(), 16) % (2**32)
seed2 = int(hashlib.md5(f'Todd_borderline_software_engineer'.encode()).hexdigest(), 16) % (2**32)
print(f'Brad seed: {seed1}')
print(f'Todd seed: {seed2}')
print('Seeds are different:', seed1 != seed2)
print()

# Compare normalized versions
cv1_norm = cv1.replace('Brad', 'NAME').replace('brad', 'name')
cv2_norm = cv2.replace('Todd', 'NAME').replace('todd', 'name')

print('=== NORMALIZED COMPARISON ===')
print(f'CV1 normalized length: {len(cv1_norm)}')
print(f'CV2 normalized length: {len(cv2_norm)}')
print(f'Normalized CVs are identical: {cv1_norm == cv2_norm}')

if cv1_norm != cv2_norm:
    print()
    print('=== DIFFERENCES ===')
    lines1 = cv1_norm.split('\n')
    lines2 = cv2_norm.split('\n')
    
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        if line1 != line2:
            print(f'Line {i+1}:')
            print(f'  Brad: {line1}')
            print(f'  Todd: {line2}')
            print()