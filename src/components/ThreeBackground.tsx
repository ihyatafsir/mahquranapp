import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

interface ParticlesProps {
    count?: number;
    isPlaying?: boolean;
}

function StarField({ count = 3000, isPlaying = false }: ParticlesProps) {
    const ref = useRef<THREE.Points>(null);

    // Generate random star positions in a sphere
    const positions = useMemo(() => {
        const pos = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            // Spherical distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = 3 + Math.random() * 7;

            pos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            pos[i * 3 + 2] = r * Math.cos(phi);
        }
        return pos;
    }, [count]);

    // Animate rotation and color
    useFrame((state) => {
        if (ref.current) {
            // Gentle rotation
            ref.current.rotation.x = state.clock.elapsedTime * 0.02;
            ref.current.rotation.y = state.clock.elapsedTime * 0.03;

            // Pulse when playing
            if (isPlaying) {
                const scale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.05;
                ref.current.scale.set(scale, scale, scale);
            }
        }
    });

    // Color based on playback state
    const color = isPlaying ? '#00ff88' : '#00ff41';

    return (
        <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
            <PointMaterial
                transparent
                color={color}
                size={0.015}
                sizeAttenuation={true}
                depthWrite={false}
                opacity={isPlaying ? 0.8 : 0.5}
                blending={THREE.AdditiveBlending}
            />
        </Points>
    );
}

// Floating geometric shapes
function FloatingGeometry({ isPlaying }: { isPlaying: boolean }) {
    const meshRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.rotation.x = state.clock.elapsedTime * 0.2;
            meshRef.current.rotation.y = state.clock.elapsedTime * 0.3;

            // Pulse effect when playing
            const breathe = isPlaying
                ? 1 + Math.sin(state.clock.elapsedTime * 3) * 0.15
                : 1 + Math.sin(state.clock.elapsedTime * 0.5) * 0.05;
            meshRef.current.scale.setScalar(breathe * 0.8);
        }
    });

    return (
        <mesh ref={meshRef} position={[0, 0, -3]}>
            <octahedronGeometry args={[1, 0]} />
            <meshBasicMaterial
                color={isPlaying ? '#00ff88' : '#00aa41'}
                wireframe
                transparent
                opacity={0.3}
            />
        </mesh>
    );
}

// Inner glow ring
function GlowRing({ isPlaying }: { isPlaying: boolean }) {
    const ringRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
        if (ringRef.current) {
            ringRef.current.rotation.z = state.clock.elapsedTime * 0.1;

            const scale = isPlaying
                ? 1 + Math.sin(state.clock.elapsedTime * 4) * 0.1
                : 1;
            ringRef.current.scale.set(scale, scale, 1);
        }
    });

    return (
        <mesh ref={ringRef} position={[0, 0, -5]}>
            <ringGeometry args={[2, 2.1, 64]} />
            <meshBasicMaterial
                color="#00ff41"
                transparent
                opacity={isPlaying ? 0.4 : 0.15}
                side={THREE.DoubleSide}
            />
        </mesh>
    );
}

interface ThreeBackgroundProps {
    isPlaying?: boolean;
}

export default function ThreeBackground({ isPlaying = false }: ThreeBackgroundProps) {
    return (
        <div
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: -1,
                background: 'radial-gradient(ellipse at center, #0a1a0a 0%, #050a05 50%, #020502 100%)',
            }}
        >
            <Canvas
                camera={{ position: [0, 0, 5], fov: 60 }}
                style={{ background: 'transparent' }}
            >
                <ambientLight intensity={0.1} />
                <StarField count={2500} isPlaying={isPlaying} />
                <FloatingGeometry isPlaying={isPlaying} />
                <GlowRing isPlaying={isPlaying} />
            </Canvas>
        </div>
    );
}
