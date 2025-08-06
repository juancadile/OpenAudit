"""
Performance Optimization Module for OpenAudit

Provides caching, parallel execution, and memory optimization
features to improve analysis performance and scalability.
"""

import gc
import hashlib
import json
import logging
import multiprocessing as mp
import pickle  # nosec B403 - Used only for internal caching of trusted data
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Track performance metrics"""

    execution_time: float
    memory_used: float
    cache_hits: int
    cache_misses: int
    parallel_workers: int
    total_tasks: int


class AnalysisCache:
    """
    Intelligent caching system for analysis results

    Supports different cache backends and automatic invalidation
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_memory_cache: int = 100,
        ttl_seconds: int = 3600,
    ):
        self.cache_dir = cache_dir or Path.home() / ".openaudit" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_cache = max_memory_cache
        self.ttl_seconds = ttl_seconds

        # In-memory cache for fast access
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {"hits": 0, "misses": 0}

        # Thread lock for thread safety
        self._lock = threading.Lock()

    def _generate_cache_key(
        self, module_name: str, data_hash: str, parameters: Dict[str, Any]
    ) -> str:
        """Generate a unique cache key"""
        # Create deterministic hash from parameters
        param_str = json.dumps(parameters, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode(), usedforsecurity=False).hexdigest()

        return f"{module_name}_{data_hash}_{param_hash}"

    def _hash_data(self, data: Any) -> str:
        """Generate hash for input data"""
        if hasattr(data, "to_dict"):
            # DataFrame-like object
            data_str = str(data.to_dict())
        elif hasattr(data, "__iter__") and not isinstance(data, str):
            # List-like object
            data_str = str([str(item) for item in data])
        else:
            data_str = str(data)

        return hashlib.md5(data_str.encode(), usedforsecurity=False).hexdigest()

    def get(
        self, module_name: str, data: Any, parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached result if available"""
        data_hash = self._hash_data(data)
        cache_key = self._generate_cache_key(module_name, data_hash, parameters)

        with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                cached_item = self.memory_cache[cache_key]

                # Check TTL
                if time.time() - cached_item["timestamp"] < self.ttl_seconds:
                    self.cache_stats["hits"] += 1
                    logger.debug(f"Cache hit (memory): {cache_key}")
                    return cached_item["result"]
                else:
                    # Expired - remove from memory cache
                    del self.memory_cache[cache_key]

            # Check disk cache
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        # NOTE: pickle.load is safe here as we only load files we created
                        # This cache system only deserializes our own trusted data
                        cached_item = pickle.load(f)  # nosec B301

                    # Check TTL
                    if time.time() - cached_item["timestamp"] < self.ttl_seconds:
                        # Load into memory cache if space available
                        if len(self.memory_cache) < self.max_memory_cache:
                            self.memory_cache[cache_key] = cached_item

                        self.cache_stats["hits"] += 1
                        logger.debug(f"Cache hit (disk): {cache_key}")
                        return cached_item["result"]
                    else:
                        # Expired - remove file
                        cache_file.unlink()

                except Exception as e:
                    logger.warning(f"Error loading cache file {cache_file}: {e}")

            self.cache_stats["misses"] += 1
            return None

    def set(
        self,
        module_name: str,
        data: Any,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
    ):
        """Store result in cache"""
        data_hash = self._hash_data(data)
        cache_key = self._generate_cache_key(module_name, data_hash, parameters)

        cached_item = {
            "result": result,
            "timestamp": time.time(),
            "module_name": module_name,
            "parameters": parameters,
        }

        with self._lock:
            # Store in memory cache
            if len(self.memory_cache) >= self.max_memory_cache:
                # Remove oldest item
                oldest_key = min(
                    self.memory_cache.keys(),
                    key=lambda k: self.memory_cache[k]["timestamp"],
                )
                del self.memory_cache[oldest_key]

            self.memory_cache[cache_key] = cached_item

            # Store in disk cache
            try:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                with open(cache_file, "wb") as f:
                    pickle.dump(cached_item, f)
                logger.debug(f"Cached result: {cache_key}")
            except Exception as e:
                logger.warning(f"Error saving cache file: {e}")

    def clear(self):
        """Clear all caches"""
        with self._lock:
            self.memory_cache.clear()

            # Remove cache files
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_file}: {e}")

            self.cache_stats = {"hits": 0, "misses": 0}
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.pkl"))),
        }


class ParallelExecutor:
    """
    Parallel execution manager for running multiple analysis modules
    or processing large datasets in parallel
    """

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.performance_metrics = PerformanceMetrics(
            execution_time=0,
            memory_used=0,
            cache_hits=0,
            cache_misses=0,
            parallel_workers=self.max_workers,
            total_tasks=0,
        )

    def execute_modules_parallel(
        self, modules: List[Any], data: Any, **kwargs
    ) -> Dict[str, Any]:
        """Execute multiple analysis modules in parallel"""
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss

        results = {}
        total_tasks = len(modules)

        logger.info(
            f"Starting parallel execution of {total_tasks} modules with {self.max_workers} workers"
        )

        # Use ThreadPoolExecutor for I/O bound operations (API calls)
        # Use ProcessPoolExecutor for CPU bound operations (computations)
        executor_type = kwargs.get("executor_type", "thread")

        if executor_type == "process":
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor

        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_module = {
                executor.submit(
                    self._execute_single_module, module, data, **kwargs
                ): module
                for module in modules
            }

            # Collect results as they complete
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                module_name = getattr(module, "__name__", str(module))

                try:
                    result = future.result()
                    results[module_name] = result
                    logger.debug(f"Module {module_name} completed successfully")
                except Exception as e:
                    logger.error(f"Module {module_name} failed: {str(e)}")
                    results[module_name] = {"error": str(e), "success": False}

        # Update performance metrics
        execution_time = time.time() - start_time
        final_memory = psutil.Process().memory_info().rss
        memory_used = (final_memory - initial_memory) / 1024 / 1024  # MB

        self.performance_metrics.execution_time = execution_time
        self.performance_metrics.memory_used = memory_used
        self.performance_metrics.total_tasks = total_tasks

        logger.info(
            f"Parallel execution completed in {execution_time:.2f}s using {memory_used:.1f}MB"
        )

        return results

    def _execute_single_module(
        self, module: Any, data: Any, **kwargs
    ) -> Dict[str, Any]:
        """Execute a single module (used by parallel executor)"""
        try:
            if hasattr(module, "analyze"):
                return module.analyze(data, **kwargs)
            elif callable(module):
                return module(data, **kwargs)
            else:
                raise ValueError(f"Module {module} is not callable")
        except Exception as e:
            logger.exception(f"Error executing module {module}")
            return {"error": str(e), "success": False}

    def process_chunks_parallel(
        self, data: List[Any], chunk_size: int, processor_func: Callable, **kwargs
    ) -> List[Any]:
        """Process data in parallel chunks"""
        # Split data into chunks
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        logger.info(
            f"Processing {len(data)} items in {len(chunks)} chunks of size {chunk_size}"
        )

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(processor_func, chunk, **kwargs): i
                for i, chunk in enumerate(chunks)
            }

            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    chunk_result = future.result()
                    results.extend(chunk_result)
                    logger.debug(f"Chunk {chunk_index} processed successfully")
                except Exception as e:
                    logger.error(f"Chunk {chunk_index} failed: {str(e)}")

        return results

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics from last execution"""
        return self.performance_metrics


class MemoryOptimizer:
    """
    Memory optimization utilities for handling large datasets
    and preventing memory leaks
    """

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident set size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual memory size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    @staticmethod
    def optimize_dataframe(df) -> Any:
        """Optimize pandas DataFrame memory usage"""
        if not hasattr(df, "dtypes"):
            return df

        optimized_df = df.copy()

        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype

            if col_type != "object":
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()

                if str(col_type)[:3] == "int":
                    if c_min > -128 and c_max < 127:
                        optimized_df[col] = optimized_df[col].astype("int8")
                    elif c_min > -32768 and c_max < 32767:
                        optimized_df[col] = optimized_df[col].astype("int16")
                    elif c_min > -2147483648 and c_max < 2147483647:
                        optimized_df[col] = optimized_df[col].astype("int32")
                elif str(col_type)[:5] == "float":
                    optimized_df[col] = optimized_df[col].astype("float32")

        return optimized_df

    @staticmethod
    def cleanup_memory():
        """Force garbage collection and cleanup"""
        gc.collect()

    @staticmethod
    def memory_monitor(func):
        """Decorator to monitor memory usage of functions"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            initial_memory = MemoryOptimizer.get_memory_usage()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                final_memory = MemoryOptimizer.get_memory_usage()
                memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]

                if memory_increase > 100:  # Log if increase > 100MB
                    logger.warning(
                        f"Function {func.__name__} used {memory_increase:.1f}MB additional memory"
                    )
                else:
                    logger.debug(
                        f"Function {func.__name__} used {memory_increase:.1f}MB additional memory"
                    )

        return wrapper


# Global cache instance
_global_cache: Optional[AnalysisCache] = None


def get_global_cache() -> AnalysisCache:
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = AnalysisCache()
    return _global_cache


def cached_analysis(cache_enabled: bool = True):
    """Decorator to cache analysis results"""

    def decorator(func):
        @wraps(func)
        def wrapper(self, data, **kwargs):
            if not cache_enabled:
                return func(self, data, **kwargs)

            cache = get_global_cache()
            module_name = getattr(self, "__class__", type(self)).__name__

            # Try to get from cache
            cached_result = cache.get(module_name, data, kwargs)
            if cached_result is not None:
                logger.debug(f"Using cached result for {module_name}")
                return cached_result

            # Execute function and cache result
            start_time = time.time()
            result = func(self, data, **kwargs)
            execution_time = time.time() - start_time

            # Add timing to metadata
            if isinstance(result, dict) and "metadata" in result:
                result["metadata"]["execution_time"] = execution_time
                result["metadata"]["cached"] = False

            # Cache the result
            cache.set(module_name, data, kwargs, result)
            logger.debug(
                f"Cached result for {module_name} (execution time: {execution_time:.2f}s)"
            )

            return result

        return wrapper

    return decorator


def performance_monitor(func):
    """Decorator to monitor performance of functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        initial_memory = MemoryOptimizer.get_memory_usage()

        try:
            result = func(*args, **kwargs)

            # Add performance metrics if result is a dict
            if isinstance(result, dict):
                execution_time = time.time() - start_time
                final_memory = MemoryOptimizer.get_memory_usage()

                performance_data = {
                    "execution_time": execution_time,
                    "memory_used_mb": final_memory["rss_mb"] - initial_memory["rss_mb"],
                    "function_name": func.__name__,
                }

                if "metadata" not in result:
                    result["metadata"] = {}
                result["metadata"]["performance"] = performance_data

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.2f}s: {str(e)}"
            )
            raise

    return wrapper


class PerformanceProfiler:
    """
    Advanced performance profiling for analysis operations
    """

    def __init__(self):
        self.profiles = {}

    def profile_analysis(
        self, analysis_func: Callable, *args, **kwargs
    ) -> Dict[str, Any]:
        """Profile an analysis function with detailed metrics"""
        import cProfile
        import io
        import pstats

        # Create profiler
        profiler = cProfile.Profile()

        # Profile the function
        start_time = time.time()
        initial_memory = MemoryOptimizer.get_memory_usage()

        profiler.enable()
        try:
            analysis_func(*args, **kwargs)
        finally:
            profiler.disable()

        execution_time = time.time() - start_time
        final_memory = MemoryOptimizer.get_memory_usage()

        # Generate profiling stats
        stats_buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_buffer)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        profile_data = {
            "execution_time": execution_time,
            "memory_used_mb": final_memory["rss_mb"] - initial_memory["rss_mb"],
            "profiling_stats": stats_buffer.getvalue(),
            "function_name": getattr(analysis_func, "__name__", str(analysis_func)),
        }

        # Store profile for later analysis
        self.profiles[analysis_func.__name__] = profile_data

        return profile_data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all performance profiles"""
        if not self.profiles:
            return {"message": "No profiles collected"}

        total_time = sum(p["execution_time"] for p in self.profiles.values())
        total_memory = sum(p["memory_used_mb"] for p in self.profiles.values())

        return {
            "total_functions_profiled": len(self.profiles),
            "total_execution_time": total_time,
            "total_memory_used_mb": total_memory,
            "average_execution_time": total_time / len(self.profiles),
            "average_memory_used_mb": total_memory / len(self.profiles),
            "profiles": {
                name: {
                    "execution_time": p["execution_time"],
                    "memory_used_mb": p["memory_used_mb"],
                }
                for name, p in self.profiles.items()
            },
        }


# Global instances
_global_executor: Optional[ParallelExecutor] = None
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_executor() -> ParallelExecutor:
    """Get or create global parallel executor"""
    global _global_executor
    if _global_executor is None:
        _global_executor = ParallelExecutor()
    return _global_executor


def get_global_profiler() -> PerformanceProfiler:
    """Get or create global performance profiler"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler
